import logging
import torch
import numpy as np

from agent.learner.grasp_script import GraspScript


class GraspScriptFlip(GraspScript):
    """
    For flipped observation, flip px.
    """
    def __init__(self, cfg):
        super().__init__(cfg)


    def forward(self, obs, extra=None, **kwargs):
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

            # Flip py if obs flipped
            scaling = extra[n]
            if scaling > 1:
                px_single = W - px_single
            
            # Add randomness
            py_single += self.rng.integers(-self.pixel_perturb, self.pixel_perturb+1, size=1)
            px_single += self.rng.integers(-self.pixel_perturb, self.pixel_perturb+1, size=1)
            py_single = np.clip(py_single, 0, H-1)
            px_single = np.clip(px_single, 0, W-1)

            # Append
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
