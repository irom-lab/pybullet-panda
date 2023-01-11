import numpy as np

from panda_gym.grasp_env import GraspEnv


class GraspFlipEnv(GraspEnv):
    def __init__(
        self,
        task=None,
        render=False,
        camera_param=None,
        #
        mu=0.5,
        sigma=0.03,
        x_offset=0.5,
        grasp_z_offset=-0.03,
    ):
        """
        """
        super(GraspFlipEnv, self).__init__(
            task=task,
            render=render,
            camera_param=camera_param,
            mu=mu,
            sigma=sigma,
            x_offset=x_offset,
            grasp_z_offset=grasp_z_offset,
        )


    def get_overhead_obs(self, camera_param):
        out = super().get_overhead_obs(camera_param)
        
        # Flip all channel from left to right for larger object
        if self.task['global_scaling'] > 1:
            out = np.flip(out, axis=2)
        return out  # uint8
