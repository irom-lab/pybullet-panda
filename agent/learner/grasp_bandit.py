import os
import logging
import torch
from torch.optim import Adam
from torch.optim import lr_scheduler
import numpy as np

from network.fcn import FCN
from util.image import rotate_tensor
from util.network import save_model


class GraspBandit():
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
        self.gradient_clip = cfg.gradient_clip

        # Backpropagating only though the label pixel or not
        self.flag_backpropagate_label_pixel_only = cfg.backpropagate_label_pixel_only

        # Parameters - grasping
        self.num_theta = cfg.num_theta
        self.thetas = torch.from_numpy(np.linspace(0, 1, num=cfg.num_theta, endpoint=False) * np.pi)
        self.min_z = cfg.min_depth
        self.max_z = cfg.max_depth

        # Parameters - pixel to xy conversion - hf for half dimension of the image in the world
        self.p2x = np.linspace(cfg.hf, -cfg.hf, num=cfg.img_w, endpoint=True)
        self.p2y = np.linspace(-cfg.hf, cfg.hf, num=cfg.img_h, endpoint=True)


    def parameters(self):
        return self.fcn.parameters()


    def build_network(self, cfg, build_optimizer=True, verbose=True):
        img_size = cfg.img_h
        assert img_size == cfg.img_w
        self.fcn = FCN(inner_channels=cfg.inner_channel_size,
                       in_channels=cfg.in_channels,
                       out_channels=cfg.out_channels,
                       img_size=img_size).to(self.device)
        if verbose:
            logging.info('Total parameters in FCN: {}'.format(
                sum(p.numel() for p in self.fcn.parameters()
                    if p.requires_grad)))

        # Load weights if provided
        if hasattr(cfg, 'network_path') and cfg.network_path is not None:
            self.load_network(cfg.network_path)

        # Create optimizer
        # if not self.eval and build_optimizer:
        #     logging.info("Build optimizer for inference.")
        #     self.build_optimizer()

        # # Load optimizer if specified
        # if hasattr(cfg, 'optimizer_path') and cfg.optimizer_path is not None:
        #     self.load_optimizer_state(torch.load(cfg.optimizer_path,  map_location=self.device))
        #     logging.info('Loaded optimizer for FCN!')


    def build_optimizer(self):
        self.optimizer = Adam(self.fcn.parameters(), lr=self.lr)    # no weight decay
        if self.lr_schedule:
            self.optimizer_scheduler = lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.lr_period,
                gamma=self.lr_decay)
        self.criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')   # normalizes FCN output


    def forward(self, obs, extra=None, flag_random=False):
        # Assume depth only
        assert obs.shape[1] == 1
        N,_,H,W = obs.shape
        
        # Set to eval mode for batchnorm
        self.fcn.eval()

        # Rotate obs to different thetas (yaw angles) and add to batch dimension
        state_rot_all = torch.empty((N, 0, H, W)).to(obs.device)
        for theta in self.thetas:
            state_rotated = rotate_tensor(obs, theta=theta)
            state_rot_all = torch.cat((state_rot_all, state_rotated), dim=1)
        state_rot_all = state_rot_all.view(N*self.num_theta, 1, H, W)

        # Random sampling pixel or forward pass
        if self.eval or not flag_random:
            fcn_pred = self(state_rot_all, extra=extra)
            # Uncombine N and C
            fcn_pred = fcn_pred.view(N, self.num_theta, H, W).cpu().numpy()
    
            # Reshape input array to a 2D array with rows being kept as with original array. Then, get idnices of max values along the columns.
            max_idx = fcn_pred.reshape(fcn_pred.shape[0],-1).argmax(1)

            # Get unravel indices corresponding to original shape of A
            (pt_all, py_all, px_all) = np.unravel_index(max_idx, fcn_pred[0,:,:,:].shape)
            # (pt, py, px) = np.unravel_index(np.argmax(fcn_pred, axis=1),
            #                                 fcn_pred.shape)  #? batch
        else:
            pt_all = self.rng.integers(0, self.num_theta, size=N)   # exlusive
            py_all = self.rng.integers(0, H, size=N)
            px_all = self.rng.integers(0, W, size=N)

        # Convert pixel to x/y
        xy_all = np.concatenate((self.p2x[px_all][None], self.p2y[py_all][None])).T
        theta_all = self.thetas[pt_all]

        # Find the target z height
        state_rot_all = state_rot_all.view(N, self.num_theta, H, W)
        norm_z = state_rot_all[torch.arange(N), pt_all, py_all, px_all].detach().cpu().numpy()
        z_all = norm_z*(self.max_z - self.min_z)    # unnormalize (min_z corresponds to max height and max_z corresponds to min height)

        # Rotate xy
        # xy_all_rotated = np.empty((0,2))
        # for xy, theta in zip(xy_all, theta_all):
        #     xy_orig = np.array([[np.cos(theta), -np.sin(theta)],
        #                         [np.sin(theta), np.cos(theta)]]).\
        #                  dot(xy)
        #     xy_all_rotated = np.vstack((xy_all_rotated, xy_orig))
        rot_all = np.concatenate((
            np.concatenate((np.cos(theta_all)[:,None,None], 
                            np.sin(-theta_all)[:,None,None]), axis=-1),
            np.concatenate((np.sin(theta_all)[:,None,None], 
                            np.cos(theta_all)[:,None,None]), axis=-1)), axis=1)
        xy_all_rotated = np.matmul(rot_all, xy_all[:,:,None])[:,:,0]
        # xy_rot = np.tile(np.array([[np.cos(theta), -np.sin(theta)],
        #                            [np.sin(theta), np.cos(theta)]])[None], 
        #                 (N, 1, 1))
        # xy_all = np.einsum('BNi,Bi->BN', xy_rot, xy_all)
        
        # Rotate pixels - since rotating around the center, first find the offset from the center with right as x and down as y
        py_offset_all = py_all - (H-1)/2
        px_offset_all = px_all - (W-1)/2
        pxy_offset_all = np.hstack((px_offset_all[:,None], py_offset_all[:,None]))
        pxy_offset_rotated_all = np.matmul(rot_all, pxy_offset_all[:,:,None])[:,:,0]
        py_rotated_all = pxy_offset_rotated_all[:,1] + (H-1)/2
        px_rotated_all = pxy_offset_rotated_all[:,0] + (W-1)/2
        py_rotated_all = np.round(py_rotated_all)
        px_rotated_all = np.round(px_rotated_all)
        
        # This is not ideal, but we have to clip since after rotation, some grasps can be out of bound if the original grasp is close to the four corners. Ideally we should not sample grasps there. For now, we can assume the object is not at the corners
        py_rotated_all = np.clip(py_rotated_all, 0, H-1)
        px_rotated_all = np.clip(px_rotated_all, 0, W-1)

        # # Debug
        # print(np.hstack((py_all[:,None], px_all[:,None])))
        # print(pxy_offset_all)
        # print(pxy_offset_rotated_all)
        # print(np.hstack((py_rotated_all[:,None], px_rotated_all[:,None])))
        # import matplotlib.pyplot as plt
        # for ind in range(7, N):
        #     theta = theta_all[ind]
        #     py = py_all[ind]
        #     px = px_all[ind]
        #     depth = obs[ind]
        #     depth_rotated = rotate_tensor(obs, theta=theta)[ind]
        #     py_rotated = int(py_rotated_all[ind])
        #     px_rotated = int(px_rotated_all[ind])
        #     print(theta, py, px, py_rotated, px_rotated)
        #     fig, axes = plt.subplots(1, 2)
        #     axes[0].imshow(depth[0].cpu().numpy())
        #     axes[0].scatter(px, py, c='r')  # axes flipped
        #     axes[1].imshow(depth_rotated[0].cpu().numpy())
        #     axes[1].scatter(px_rotated, py_rotated, c='r')
        #     axes[0].set_title('Original')
        #     axes[1].set_title('Rotated')
        #     plt.show()
        output = np.hstack((xy_all_rotated, 
                            z_all[:,None], 
                            theta_all[:,None],
                            py_all[:,None],
                            px_all[:,None],
                            py_rotated_all[:,None],
                            px_rotated_all[:,None],
                            ))
        return output


    def __call__(self, state, extra=None, verbose=False):
        return self.fcn(state, extra)


    def update(self, batch):
        # Set to training mode for batchnorm 
        self.fcn.train()
        
        # Unpack batch
        depth_train_batch, ground_truth_batch, mask_train_batch = batch

        # Forward, get loss, zero gradients
        pred_train_batch = self.fcn(depth_train_batch).squeeze(1)  # NxHxW
        train_loss = self.criterion(pred_train_batch, ground_truth_batch)
        self.optimizer.zero_grad()

        # mask gradient for non-selected pixels
        if self.flag_backpropagate_label_pixel_only:
            pred_train_batch.retain_grad()
            pred_train_batch.register_hook(lambda grad: grad * mask_train_batch)

        # Update params using clipped gradients
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.fcn.parameters(), self.gradient_clip)
        self.optimizer.step()
        return train_loss.detach().cpu().numpy()


    def update_hyper_param(self):
        pass


    def load_optimizer_state(self, ):
        raise NotImplementedError


    def save(self, step, logs_path, max_model=None):
        logging.info('Saving model at step %d', step)
        path_c = os.path.join(logs_path, 'critic')
        return save_model(self.fcn, path_c, 'critic', step, max_model)


    def remove(self, step, logs_path):
        path_c = os.path.join(logs_path, 'critic',
                              'critic-{}.pth'.format(step))
        if os.path.exists(path_c):
            os.remove(path_c)
            logging.info("Remove {}".format(path_c))
