import os
import logging
import torch
from torch.optim import Adam
from torch.optim import lr_scheduler
import numpy as np

from network.fcn import FCN
from util.image import rotate_tensor
from util.torch import save_model


class GraspBandit():
    def __init__(self, cfg):
        self.device = cfg.device
        self.eval = cfg.eval
        if not self.eval:
            self.lr = cfg.lr
            self.lr_schedule = cfg.lr_schedule
            if self.lr_schedule:
                self.lr_period = cfg.lr_period
                self.lr_decay = cfg.lr_decay
                self.lr_end = cfg.lr_end
        self.rng = np.random.default_rng(seed=cfg.seed)

        # Constants
        self.num_theta = cfg.num_theta
        self.thetas = torch.from_numpy(np.linspace(0, 1, num=cfg.num_theta, endpoint=False) * np.pi)
        self.delta_z = 0.03
        self.min_ee_z = 0.15
        self.max_obj_height = 0.05  #?

        # Pixel to xy conversion
        self.p2x = np.linspace(0.10, -0.10, num=cfg.img_w, endpoint=True)
        self.p2y = np.linspace(-0.10, 0.10, num=cfg.img_h, endpoint=True)


    def parameters(self):
        return self.fcn.parameters()


    def build_network(self, cfg, build_optimizer=True, verbose=True):
        img_size = cfg.img_h
        assert img_size == cfg.img_w
        self.fcn = FCN(inner_channels=cfg.inner_channels,
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
        self.criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')


    def forward(self, state, append=None, flag_random=False):
        # TODO: random flag for each env
        # Assume depth only
        assert state.shape[1] == 1
        N,_,H,W = state.shape

        # Rotate image
        state_rot_all = torch.empty((N, 0, H, W)).to(state.device)
        for theta in self.thetas:
            state_rotated = rotate_tensor(state, theta=theta)
            state_rot_all = torch.cat((state_rot_all, state_rotated), dim=1)
        # Combine N and C
        state_rot_all = state_rot_all.view(N*self.num_theta, 1, H, W)

        if self.eval or not flag_random:
            fcn_pred = self(state_rot_all, append=append)
            # Uncombine N and C
            fcn_pred = fcn_pred.view(N, self.num_theta, H, W).cpu().numpy()
    
            # Reshape input array to a 2D array with rows being kept as with original array. Then, get idnices of max values along the columns.
            max_idx = fcn_pred.reshape(fcn_pred.shape[0],-1).argmax(1)

            # Get unravel indices corresponding to original shape of A
            (pt, py, px) = np.unravel_index(max_idx, fcn_pred[0,:,:,:].shape)
            # (pt, py, px) = np.unravel_index(np.argmax(fcn_pred, axis=1),
            #                                 fcn_pred.shape)  #? batch
        else:
            pt = self.rng.integers(0, self.num_theta, size=N) # exlusive
            py = self.rng.integers(0, H, size=N)
            px = self.rng.integers(0, W, size=N)

        # Convert pixel to x/y
        xy_all = np.concatenate((self.p2x[px][None], self.p2y[py][None])).T 
        theta_all = self.thetas[pt]

        # Find the target z height
        state_rot_all = state_rot_all.view(N, self.num_theta, H, W)
        norm_z = state_rot_all[torch.arange(N), pt, py, px].detach().cpu().numpy()
        z_all = norm_z*0.3  # unnormalize   # TODO
        z_target_all = np.maximum(0, z_all - self.delta_z)  # clip
        z_target_ee_all = z_target_all + self.min_ee_z

        # Rotate into local frame
        xy_rot = np.tile(np.array([[np.cos(theta), -np.sin(theta)],
                                   [np.sin(theta), np.cos(theta)]])[None], 
                        (N, 1, 1))
        xy_all = np.einsum('BNi,Bi->BN', xy_rot, xy_all)
        xy_all[:,0] += 0.5
        output = np.hstack((xy_all, z_target_ee_all[:,None], theta_all[:,None], py[:,None], px[:,None]))
        return output


    def __call__(self, state, append=None, verbose=False):
        fcn_pred = self.fcn(state, append)
        # if verbose:
            # logging.info('FCN with state {}'.format(state))
        return fcn_pred


    def update(self, batch):
        depth_train_batch, ground_truth_batch, mask_train_batch = batch

        # Forward, get loss, zero gradients
        pred_train_batch = self.fcn(depth_train_batch).squeeze(1)  # NxHxW
        train_loss = self.criterion(pred_train_batch, ground_truth_batch)
        self.optimizer.zero_grad()

        # mask gradient for non-selected pixels
        pred_train_batch.retain_grad()
        pred_train_batch.register_hook(lambda grad: grad * mask_train_batch)

        # Update params using clipped gradients
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.fcn.parameters(), 10)
        self.optimizer.step()
        return train_loss.detach().cpu().numpy()


    def update_hyper_param(self):
        pass


    def load_optimizer_state(self, ):
        pass


    def save(self, step, logs_path, max_model=None):
        path_c = os.path.join(logs_path, 'critic')
        return save_model(self.fcn, path_c, 'critic', step, max_model)


    def remove(self, step, logs_path):
        path_c = os.path.join(logs_path, 'critic',
                              'critic-{}.pth'.format(step))
        # logging.info("Remove {}".format(path_c))
        if os.path.exists(path_c):
            os.remove(path_c)
