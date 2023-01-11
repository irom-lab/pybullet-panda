import os
import logging
import torch
from torch.optim import Adam
from torch.optim import lr_scheduler
import numpy as np
import torch.nn.functional as F 


from agent.algorithm.poem import cosine_similarity, calculate_action_cost_matrix, metric_fixed_point, contrastive_loss
from network.encoder import Encoder
from network.fcn import FCN
from network.mlp import MLP
from util.image import rotate_tensor
from util.network import save_model


class GraspBanditEq():
    def __init__(self, cfg):
        self.device = cfg.device
        self.rng = np.random.default_rng(seed=cfg.seed)
        self.eval = cfg.eval

        # Learning rate, schedule
        if not self.eval:
            self.lr = cfg.lr
            self.lr_schedule = cfg.lr_schedule
            if self.lr_schedule:
                self.lr_period = cfg.lr_period
                self.lr_decay = cfg.lr_decay
                self.lr_end = cfg.lr_end

        # Grdient clipping
        self.cfg_gradient_clip = cfg.gradient_clip

        # Parameters - grasping
        self.num_theta = cfg.num_theta
        self.thetas = torch.from_numpy(np.linspace(0, 1, num=cfg.num_theta, endpoint=False) * np.pi)
        self.min_z = cfg.min_depth
        self.max_z = cfg.max_depth

        # Parameters - pixel to xy conversion - hf for half dimension of the image in the world
        self.p2x = np.linspace(cfg.hf, -cfg.hf, num=cfg.img_w, endpoint=True)
        self.p2y = np.linspace(-cfg.hf, cfg.hf, num=cfg.img_h, endpoint=True)

        # Parameters - POEM
        self.action_dim = 2
        self.obs_channel_size = cfg.obs_channel_size
        self.latent_state_size = cfg.latent_state_size
        self.latent_action_size = cfg.latent_action_size
        self.cfg_poem = cfg.poem


    def parameters(self):
        return self.action_decoder.parameters() # ignore other ones for now


    def build_network(self, cfg, build_optimizer=True, verbose=True):
        img_size = cfg.img_h
        assert img_size == cfg.img_w
        
        # Action decoder - FCN - (obs, latent_action) -> affordance
        if verbose:
            logging.info('Action decoder: FCN')
        self.action_decoder = FCN(inner_channels=cfg.action_decoder.inner_channel_size,
                                  in_channels=self.obs_channel_size+self.latent_action_size,
                                  out_channels=1,
                                  img_size=img_size).to(self.device)
        
        # State encoder - CNN+MLP - obs -> latent_state
        if verbose:
            logging.info('State encoder: Encoder (CNN+MLP)')
        self.state_encoder = Encoder(in_channels=self.obs_channel_size,
                                     img_sz=img_size,
                                     kernel_sz=cfg.state_encoder.kernel_size,
                                     stride=cfg.state_encoder.stride,
                                     padding=cfg.state_encoder.padding,
                                     n_channel=cfg.state_encoder.num_channel,
                                     mlp_hidden_dim=cfg.state_encoder.mlp_hidden_dim,
                                     mlp_output_dim=self.latent_state_size,
                                     use_sm=False,
                                     use_spec=False,
                                     use_bn=False,
                                     device=self.device,
                                     verbose=True)
        
        # Latent policy - MLP - latent_state -> latent_action
        if verbose:
            logging.info('Latent policy: MLP')
        self.latent_policy = MLP(layer_size=[self.latent_state_size,
                                             *cfg.latent_policy.hidden_dim,
                                             self.latent_action_size]
                                ).to(self.device)
        
        # Action encoder - CNN+MLP - (obs, action) -> latent_action
        if verbose:
            logging.info('Action encoder: Encoder (CNN+MLP)')
        self.action_encoder = Encoder(in_channels=self.obs_channel_size+self.action_dim,
                                      img_sz=img_size,
                                      kernel_sz=cfg.action_encoder.kernel_size,
                                      stride=cfg.action_encoder.stride,
                                      padding=cfg.action_encoder.padding,
                                      n_channel=cfg.action_encoder.num_channel,
                                      mlp_hidden_dim=cfg.action_encoder.mlp_hidden_dim,
                                      mlp_output_dim=self.latent_action_size,
                                      use_sm=False,
                                      use_spec=False,
                                      use_bn=False,
                                      device=self.device,
                                      verbose=True)
        # TODO: tie weights of the CNN in action_encoder and state_encoder
        # # Action encoder - MLP - action -> latent_action
        # self.action_encoder = MLP(layer_size=[2,    # py, px
        #                                      *self.action_encoder.hidden_dim,
        #                                      self.latent_action_size]
        #                           ).to(self.device)

        # Load weights if provided
        if hasattr(cfg, 'network_path') and cfg.network_path is not None:
            self.load_network(cfg.network_path.action_decoder)
            self.load_network(cfg.network_path.state_encoder)
            self.load_network(cfg.network_path.latent_policy)
            self.load_network(cfg.network_path.action_encoder)
        if verbose:
            logging.info('Total parameters in Action Decoder (FCN): {}'.format(
                sum(p.numel() for p in self.action_decoder.parameters() if p.requires_grad)))
            logging.info('Total parameters in State Encoder (CNN+MLP): {}'.format(
                sum(p.numel() for p in self.state_encoder.parameters() if p.requires_grad)))
            logging.info('Total parameters in Latent Policy (MLP): {}'.format(
                sum(p.numel() for p in self.latent_policy.parameters() if p.requires_grad)))
            logging.info('Total parameters in Action Encoder (CNN+MLP): {}'.format(
                sum(p.numel() for p in self.action_encoder.parameters() if p.requires_grad)))

        # Create optimizer
        if not self.eval and build_optimizer:
            logging.info("Build optimizer for inference.")
            self.build_optimizer()

        # Load optimizer if specified
        if hasattr(cfg, 'optimizer_path') and cfg.optimizer_path is not None:
            self.load_optimizer_state(torch.load(cfg.optimizer_path.state, 
                                                 map_location=self.device))
            logging.info('Loaded optimizer for FCN!')


    def build_optimizer(self):
        self.action_decoder_optimizer = Adam(self.action_decoder.parameters(), 
                                             lr=self.lr.action_decoder)
        self.state_encoder_optimizer = Adam(self.state_encoder.parameters(), 
                                            lr=self.lr.state_encoder)
        self.latent_policy_optimizer = Adam(self.latent_policy.parameters(),
                                            lr=self.lr.latent_policy)
        self.action_encoder_optimizer = Adam(self.action_encoder.parameters(), 
                                             lr=self.lr.action_encoder)
        if self.lr_schedule:
            raise NotImplementedError
            # self.optimizer_scheduler = lr_scheduler.StepLR(
                # self.optimizer,
                # step_size=self.lr_period,
                # gamma=self.lr_decay)

        # Cross entropy loss - for latent policy training and action decoder training
        self.ce_loss = torch.nn.BCEWithLogitsLoss(reduction='mean')
        # Contrastive loss is defined manually without using criterion


    def forward(self, obs, append=None, flag_random=False):
        # Assume depth only
        assert obs.shape[1] == 1
        N,_,H,W = obs.shape

        # Rotate obs to different thetas (yaw angles) and add to batch dimension
        state_rot_all = torch.empty((N, 0, H, W)).to(obs.device)
        for theta in self.thetas:
            state_rotated = rotate_tensor(obs, theta=theta)
            state_rot_all = torch.cat((state_rot_all, state_rotated), dim=1)
        state_rot_all = state_rot_all.view(N*self.num_theta, 1, H, W)

        # Random sampling pixel or forward pass
        if self.eval or not flag_random:

            # Pass obs through state encoder to get latent state
            latent_state = self.state_encoder(state_rot_all)
            
            # Pass latent state through latent policy to get latent action
            latent_action = self.latent_policy(latent_state)
            
            #! not normalizing latent action from latent policy right now

            # Add latent action to channel dimension of obs
            latent_action = latent_action.unsqueeze(-1).unsqueeze(-1)
            latent_action = latent_action.repeat(1, 1, H, W)
            obs_with_latent_action = torch.cat((state_rot_all, latent_action), dim=1)

            # Pass through action decoder to get affordance map
            fcn_pred = self.action_decoder(obs_with_latent_action)

            # Unfold N and num_theta
            fcn_pred = fcn_pred.view(N, self.num_theta, H, W).cpu().numpy()

            # Reshape input array to a 2D array with rows being kept as with original array. Then, get indices of max values along the columns.
            max_idx = fcn_pred.reshape(fcn_pred.shape[0],-1).argmax(1)

            # Get unravel indices corresponding to original shape of A
            (pt, py, px) = np.unravel_index(max_idx, fcn_pred[0,:,:,:].shape)
        else:
            pt = self.rng.integers(0, self.num_theta, size=N)   # exlusive
            py = self.rng.integers(0, H, size=N)
            px = self.rng.integers(0, W, size=N)

        # Convert pixel to x/y
        xy_all = np.concatenate((self.p2x[px][None], self.p2y[py][None])).T 
        theta_all = self.thetas[pt]

        # Find the target z height
        state_rot_all = state_rot_all.view(N, self.num_theta, H, W)
        norm_z = state_rot_all[torch.arange(N), pt, py, px].detach().cpu().numpy()
        z_all = norm_z*(self.max_z - self.min_z)    # unnormalize (min_z corresponds to max height and max_z corresponds to min height)

        # Rotate into local frame
        # logging.info(f'xy before rotation: {xy_all[:2]}')
        # logging.info(f'theta_all: {theta_all[:2]}')
        xy_all_rotated = np.empty((0,2))
        for xy, theta in zip(xy_all, theta_all):
            xy_orig = np.array([[np.cos(theta), -np.sin(theta)],
                                [np.sin(theta), np.cos(theta)]]).\
                         dot(xy)
            xy_all_rotated = np.vstack((xy_all_rotated, xy_orig))
        # xy_rot = np.tile(np.array([[np.cos(theta), -np.sin(theta)],
        #                            [np.sin(theta), np.cos(theta)]])[None], 
        #                 (N, 1, 1))
        # xy_all = np.einsum('BNi,Bi->BN', xy_rot, xy_all)
        # logging.info(f'xy after rotation: {xy_all_rotated[:2]}')
        output = np.hstack((xy_all_rotated, 
                            z_all[:,None], 
                            theta_all[:,None],
                            py[:,None], 
                            px[:,None]))
        return output


    # def __call__(self, state, append=None, verbose=False):
    #     fcn_pred = self.fcn(state, append)
    #     # if verbose:
    #         # logging.info('FCN with state {}'.format(state))
    #     return fcn_pred


    # def update(self, batch):
    #     depth_train_batch, ground_truth_batch, mask_train_batch = batch

    #     # Encoder training
    #     encoded_state_batch = self.encoder(depth_train_batch)   # Nx1xHxW
    #     latent_action_batch = self.policy(encoded_state_batch)  # Nx1xHxW
        
    #     # Forward, get loss, zero gradients
    #     pred_train_batch = self.fcn(depth_train_batch).squeeze(1)  # NxHxW
    #     train_loss = self.ce_loss(pred_train_batch, ground_truth_batch)
    #     self.optimizer.zero_grad()

    #     # mask gradient for non-selected pixels
    #     pred_train_batch.retain_grad()
    #     pred_train_batch.register_hook(lambda grad: grad * mask_train_batch)

    #     # Update params using clipped gradients
    #     train_loss.backward()
    #     torch.nn.utils.clip_grad_norm_(self.fcn.parameters(), 10)
    #     self.optimizer.step()

    #     return train_loss.detach().cpu().numpy()


    def update_action_decoder(self, batch):
        """
        Update the action decoder with ce loss. 
        Do not update the action encoder with ce loss.
        """

        # Turn off action encoder gradient and turn on action decoder gradient
        for param in self.action_encoder.parameters():
            param.requires_grad = False
        for param in self.action_decoder.parameters():
            param.requires_grad = True

        # Extract from batch
        obs_batch, ground_truth_batch, mask_batch, action_batch, _ = batch

        # Add action to channel dimension of obs
        _, _, H, W = obs_batch.shape
        action_batch = action_batch.unsqueeze(-1).unsqueeze(-1)
        action_batch = action_batch.repeat(1, 1, H, W)
        obs_with_action = torch.cat((obs_batch, action_batch), dim=1)

        # Pass obs and action through action_encoder
        latent_action = self.action_encoder(obs_with_action)

        #! not normalizing latent action from action encoder right now

        # Add latent action to channel dimension of obs
        latent_action = latent_action.unsqueeze(-1).unsqueeze(-1)
        latent_action = latent_action.repeat(1, 1, H, W)
        obs_with_latent_action = torch.cat((obs_batch, latent_action), dim=1)
        # TODO: gradient issue with depth_batch?
        
        # Pass through action decoder to get affordance map
        pred_train_batch = self.action_decoder(obs_with_latent_action)
        
        # Forward, get loss, zero gradients
        action_decoder_training_loss = self.criterion(pred_train_batch, ground_truth_batch)
        self.action_decoder_optimizer.zero_grad()
        self.action_encoder_optimizer.zero_grad()

        # mask gradient for non-selected pixels
        pred_train_batch.retain_grad()
        pred_train_batch.register_hook(lambda grad: grad * mask_batch)

        # Update action decoder paramaters using clipped gradients
        action_decoder_training_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.action_decoder.parameters(), 
                                       self.cfg_gradient_clip.action_decoder)
        self.action_decoder_optimizer.step()
        return action_decoder_training_loss.detach().cpu().numpy()


    def update_latent_policy(self, batch):
        """
        Update the state encoder and the latent policy with ce loss.
        Update the state encoder and action encoder with alignment loss. 
        Do not update the action decoder with ce loss.
        """

        # Turn on action encoder gradient and turn off action decoder gradient
        for param in self.action_encoder.parameters():
            param.requires_grad = True
        for param in self.action_decoder.parameters():
            param.requires_grad = False

        # Extract from batch
        obs_batch, ground_truth_batch, mask_batch, action_batch, reward_batch = batch
        _, _, H, W = obs_batch.shape

        ################### ce loss ###################

        # Pass obs through state encoder
        latent_state = self.state_encoder(obs_batch)    # N x latent_state_size
        
        # Pass latent state through latent policy to get latent action
        latent_action = self.latent_policy(latent_state)  # N x latent_action_size

        # Add latent action to channel dimension of obs
        latent_action = latent_action.unsqueeze(-1).unsqueeze(-1)
        latent_action = latent_action.repeat(1, 1, H, W)
        obs_with_latent_action = torch.cat((obs_batch, latent_action), dim=1)
        
        # Pass obs and latent action through action decoder
        pred_train_batch = self.action_decoder(obs_with_latent_action)
        
        # Get cross entropy loss
        ce_loss = self.criterion(pred_train_batch, ground_truth_batch)

        ################### alignment loss ###################
    
        # Get indices of batch where reward is 1
        indices = torch.where(reward_batch == 1)[0]
        # split indices into two groups randomly, with equal number of elements
        indices_1, indices_2 = torch.split(indices, len(indices) // 2)
        # get obs and actions for each group
        obs_1 = obs_batch[indices_1]
        obs_2 = obs_batch[indices_2]
        actions_1 = action_batch[indices_1]
        actions_2 = action_batch[indices_2] 

        # Get latent state and then similarity matrix
        latent_state_1 = self.state_encoder(obs_1)
        latent_state_2 = self.state_encoder(obs_2)
        similarity_matrix = cosine_similarity(latent_state_1, latent_state_2)

        # Add action to channel dimension of obs
        actions_1 = actions_1.unsqueeze(-1).unsqueeze(-1)
        actions_1 = actions_1.repeat(1, 1, H, W)
        actions_2 = actions_2.unsqueeze(-1).unsqueeze(-1)
        actions_2 = actions_2.repeat(1, 1, H, W)
        obs_with_action_1 = torch.cat((obs_1, actions_1), dim=1)
        obs_with_action_2 = torch.cat((obs_2, actions_2), dim=1)

        # Get latent action using action encoder
        latent_action_1 = self.action_encoder(obs_with_action_1)
        latent_action_2 = self.action_encoder(obs_with_action_2)

        # Normalize each latent action 1 and 2 to have norm=1
        #! right now we are treating latent action as a continuous vector, instead of logits in the jumping task where we normalize them with softmax
        #! TODO: check latent action distances
        latent_action_1_normalized = F.normalize(latent_action_1, p=2, dim=1)
        latent_action_2_normalized = F.normalize(latent_action_2, p=2, dim=1)

        # Get metric values - no ground_truth or use_bisim right now
        cost_matrix = calculate_action_cost_matrix(latent_action_1_normalized, 
                                                   latent_action_2_normalized, 
                                                   hardcode_encoder=self.flag_use_pse)
        # metric_vals = metric_fixed_point(cost_matrix, self.poem_gamma, device=self.device)
        metric_vals = cost_matrix   # since grasping is bandit

        # Get contrastive loss
        alignment_loss = contrastive_loss(similarity_matrix,
                                          metric_vals,
                                          self.cfg_poem.temperature,
                                          coupling_temperature=self.cfg_poem.coupling_temperature,
                                          use_coupling_weights=self.cfg_poem.use_coupling_weights) 

        ################### TODO: try L1 loss ###################
		# l1_loss = compute_l1_loss(action_encoder, 
            # optimal_data_tuple_batch, learned_encoder_coefficient=FLAGS.learned_encoder_coefficient, device=device)

        ################### update ###################
        combined_loss = ce_loss + self.cfg_poem.alignment_loss_weight*alignment_loss 

        # Zero gradients
        self.state_encoder.zero_grad()
        self.latent_policy.zero_grad()
        self.action_encoder.zero_grad()
        self.action_decoder.zero_grad()

        # mask gradient for non-selected pixels
        pred_train_batch.retain_grad()
        pred_train_batch.register_hook(lambda grad: grad * mask_batch)

        # Update action decoder paramaters using clipped gradients
        combined_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.state_encoder.parameters(), 
                                       self.cfg_gradient_clip.state_encoder)
        torch.nn.utils.clip_grad_norm_(self.latent_policy.parameters(), 
                                       self.cfg_gradient_clip.latent_policy)
        torch.nn.utils.clip_grad_norm_(self.action_encoder.parameters(), 
                                       self.cfg_gradient_clip.action_encoder)
        self.state_encoder_optimizer.step()
        self.latent_policy_optimizer.step()
        self.action_encoder_optimizer.step()
        return ce_loss.detach().cpu().numpy(), alignment_loss.detach().cpu().numpy()


    def update_hyper_param(self):
        pass


    def load_optimizer_state(self, ):
        raise NotImplementedError


    def save(self, step, logs_path, max_model=None):
        logging.info('Saving model at step %d', step)
        save_model(self.state_encoder, os.path.join(logs_path, 'critic'), 
                        'state_encoder', step, max_model)
        save_model(self.latent_policy, os.path.join(logs_path, 'critic'), 
                        'latent_policy', step, max_model)
        save_model(self.action_encoder, os.path.join(logs_path, 'critic'), 
                        'action_encoder', step, max_model)
        return save_model(self.action_decoder, os.path.join(logs_path, 'critic'), 
                          'action_decoder', step, max_model)


    def remove(self, step, logs_path):
        model_name = ['action_decoder', 'state_encoder', 'latent_policy', 'action_encoder']
        for name in model_name:
            path = os.path.join(logs_path, 'critic', f'{name}-{step}.pth')
            if os.path.exists(path):
                os.remove(path)
            logging.info("Remove {}".format(path))
