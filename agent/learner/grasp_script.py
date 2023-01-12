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
        self.p2x = np.linspace(cfg.hf, -cfg.hf, num=cfg.img_w, endpoint=True)
        self.p2y = np.linspace(-cfg.hf, cfg.hf, num=cfg.img_h, endpoint=True)


    def parameters(self):
        return 0


    def build_network(self, cfg, build_optimizer=True, verbose=True):
        pass


    def build_optimizer(self):
        pass


    def forward(self, obs, append=None, flag_random=False):
        # Assume depth only
        assert obs.shape[1] == 1
        N,_,H,W = obs.shape
        obs = obs.detach().cpu().numpy()[:,0]

        # Sample any pixel with positive depth
        py = np.empty((0), dtype='int')
        px = np.empty((0), dtype='int')
        for n in range(N):
            obs_single_ind = np.where(obs[n] > self.norm_z_threshold)
            ind = self.rng.integers(0, len(obs_single_ind[0]), size=1)
            py_single = obs_single_ind[0][ind].astype('int')
            px_single = obs_single_ind[1][ind].astype('int')
            py = np.append(py, py_single)
            px = np.append(px, px_single)

        # Sample random theta
        pt = self.rng.integers(0, self.num_theta, size=N)   # exclusive

        # Convert pixel to x/y
        xy_all = np.concatenate((self.p2x[px][None], self.p2y[py][None])).T
        theta_all = self.thetas[pt]

        # Find the target z height
        norm_z = obs[np.arange(N), py, px]
        z_all = norm_z*(self.max_z - self.min_z)    # unnormalize (min_z corresponds to max height and max_z corresponds to min height)

        # Rotate into local frame
        rot_all = np.concatenate((
            np.concatenate((np.cos(theta_all)[:,None,None], 
                            np.sin(-theta_all)[:,None,None]), axis=-1),
            np.concatenate((np.sin(theta_all)[:,None,None], 
                            np.cos(theta_all)[:,None,None]), axis=-1)), axis=1)
        xy_all_rotated = np.matmul(rot_all, xy_all[:,:,None])[:,:,0]
        output = np.hstack((xy_all_rotated, 
                            z_all[:,None], 
                            theta_all[:,None],
                            py[:,None], 
                            px[:,None]))
        return output


    def __call__(self, state, append=None, verbose=False):
        return self.forward(state, append)


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
