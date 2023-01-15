import logging
import torch
import numpy as np


class GraspScript():
    """
    Randomly sample a pixel where the normalized depth value (from env) is above the norm_z_threshold. Randomly sample theta from discretized ones.
    """
    def __init__(self, cfg):
        self.device = cfg.device
        self.rng = np.random.default_rng(seed=cfg.seed)

        # Parameters - grasping
        self.num_theta = cfg.num_theta
        self.thetas = torch.from_numpy(np.linspace(0, 1, num=cfg.num_theta, endpoint=False) * np.pi)
        self.min_z = cfg.min_depth
        self.max_z = cfg.max_depth
        self.norm_z_threshold = cfg.norm_z_threshold

        # Parameters - pixel to xy conversion - hf for half dimension of the image in the world
        # Note here the x axis is aligned with the image, but opposite of the table x-axis
        self.p2x = np.linspace(-cfg.hf, cfg.hf, num=cfg.img_w, endpoint=True)
        self.p2y = np.linspace(-cfg.hf, cfg.hf, num=cfg.img_h, endpoint=True)

        # Random perturbation
        self.pixel_perturb = cfg.pixel_perturb


    def parameters(self):
        return 0


    def build_network(self, cfg, build_optimizer=True, verbose=True):
        pass


    def build_optimizer(self):
        pass


    def forward(self, obs, extra=None, **kwargs):
        """Different from grasp_bandit, here we do not rotate observation, but just pick pixels from the original observation, and then randomly pick theta."""

        # Assume depth at first channel
        N,C,H,W = obs.shape
        assert C != 3, "Assume D or RGBD input."
        depth_channel_ind = 0
        obs = obs.detach().cpu().numpy()[:,depth_channel_ind]

        # Convert to float if uint8
        if obs.dtype == np.uint8:
            obs = obs/255.0

        # Sample any pixel with positive depth
        py_all = np.empty((0), dtype='int')
        px_all = np.empty((0), dtype='int')
        for n in range(N):

            # Flip obs back in x if scaling
            if extra is not None:   # not the best design
                table_rgba_all = extra[n]
                if table_rgba_all[0] < 0.5:
                    obs[n] = np.flip(obs[n], axis=-1)

            obs_single_ind = np.where(obs[n] > self.norm_z_threshold)
            ind = self.rng.integers(0, len(obs_single_ind[0]), size=1)
            py_single = obs_single_ind[0][ind].astype('int')
            px_single = obs_single_ind[1][ind].astype('int')

            # perturb
            py_single += self.rng.integers(-self.pixel_perturb, self.pixel_perturb+1, size=1)
            px_single += self.rng.integers(-self.pixel_perturb, self.pixel_perturb+1, size=1)
            py_single = np.clip(py_single, 0, H-1)
            px_single = np.clip(px_single, 0, W-1)

            py_all = np.append(py_all, py_single)
            px_all = np.append(px_all, px_single)

        # Sample random theta
        pt_all = self.rng.integers(0, self.num_theta, size=N)   # exclusive

        # Convert pixel to x/y, but x/y are rotated
        xy_all = np.concatenate((self.p2x[px_all][None], self.p2y[py_all][None])).T
        theta_all = self.thetas[pt_all]

        # Flip x axis so that x axis is aligned with the image x axis
        xy_all[:,0] *= -1

        # Find the target z height at the pixels
        norm_z = obs[torch.arange(N), py_all, px_all]
        z_all = norm_z*(self.max_z - self.min_z)    # unnormalize (min_z corresponds to max height and max_z corresponds to min height)

        # Rotate pixels to theta- since rotating around the center, first find the offset from the center with right as x and down as y
        rot_all = np.concatenate((
            np.concatenate((np.cos(theta_all)[:,None,None],
                            -np.sin(theta_all)[:,None,None]), axis=-1),
            np.concatenate((np.sin(theta_all)[:,None,None], 
                            np.cos(theta_all)[:,None,None]), axis=-1)), axis=1)
        py_offset_all = py_all - (H-1)/2
        px_offset_all = px_all - (W-1)/2
        pxy_offset_all = np.hstack((px_offset_all[:,None], py_offset_all[:,None]))
        pxy_rot_offset_all = np.matmul(rot_all, pxy_offset_all[:,:,None])[:,:,0]
        py_rot_all = pxy_rot_offset_all[:,1] + (H-1)/2
        px_rot_all = pxy_rot_offset_all[:,0] + (W-1)/2
        py_rot_all = np.round(py_rot_all)
        px_rot_all = np.round(px_rot_all)

        # Return all data
        return np.hstack((xy_all, 
                          z_all[:,None], 
                          theta_all[:,None],
                          py_all[:,None],
                          px_all[:,None],
                          py_rot_all[:,None],
                          px_rot_all[:,None],
                          ))


    def __call__(self, state, extra=None, verbose=False):
        return self.forward(state, extra)


    def update(self, batch):
        pass


    def update_hyper_param(self):
        pass


    def load_optimizer_state(self, ):
        raise NotImplementedError


    def save(self, step, logs_path, max_model=None):
        pass


    def remove(self, step, logs_path):
        pass
