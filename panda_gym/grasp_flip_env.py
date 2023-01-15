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
        lift_threshold=0.02,
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
            lift_threshold=lift_threshold,
        )

    def reset_task(self, task):
        super().reset_task(task)
        
        # Change color
        self._p.changeVisualShape(self._table_id, -1,
                                  rgbaColor=task.table_rgba)


    def get_overhead_obs(self, camera_param):
        out = super().get_overhead_obs(camera_param)
        
        # Flip all channel from left to right for darker table color, assume grayscale, thus only look at first channel
        if self.task.table_rgba[0] < 0.5:
            out = np.flip(out, axis=-1)

        # Flip all channel from left to right for larger object
        # if self.task['global_scaling'] > 1:
        #     out = np.flip(out, axis=2)
        
        # Add noise to normalized depth
        # out = out.astype(np.float32)/255.0
        # out += self.rng.normal(0, 0.01, out.shape)
        # out = np.clip(out, 0, 1)
        # return np.uint8(out*255)  # uint8
        return out
