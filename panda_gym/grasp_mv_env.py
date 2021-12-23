from abc import ABC
import numpy as np

from panda_gym.grasp_env import GraspEnv
from panda.util_geom import quatMult, euler2quat, euler2quat, quat2rot


class GraspMultiViewEnv(GraspEnv, ABC):
    def __init__(
        self,
        task=None,
        render=False,
        img_H=128,
        img_W=128,
        use_rgb=False,
        max_steps_train=100,
        max_steps_eval=100,
        done_type='fail',
        #
        mu=0.5,
        sigma=0.01,
        # camera_params=None,
    ):
        """
        Args:
            task (str, optional): the name of the task. Defaults to None.
            img_H (int, optional): the height of the image. Defaults to 128.
            img_W (int, optional): the width of the image. Defaults to 128.
            use_rgb (bool, optional): whether to use RGB image. Defaults to
                True.
            render (bool, optional): whether to render the environment.
                Defaults to False.
            max_steps_train (int, optional): the maximum number of steps to
                train. Defaults to 100.
            max_steps_eval (int, optional): the maximum number of steps to
                evaluate. Defaults to 100.
            done_type (str, optional): the type of the done. Defaults to
                'fail'.
        """
        super(GraspMultiViewEnv, self).__init__(
            task=task,
            render=render,
            img_H=img_H,
            img_W=img_W,
            use_rgb=use_rgb,
            max_steps_train=max_steps_train,
            max_steps_eval=max_steps_eval,
            done_type=done_type,
            mu=mu,
            sigma=sigma,
            camera_params=None,  #! use wrist view
        )

    @property
    def action_dim(self):
        """
        Dimension of robot action - x,y,yaw
        """
        return 3

    # @abstractmethod
    # def report(self):
    #     """
    #     Print information of robot dynamics and observation.
    #     """
    #     raise NotImplementedError

    # @abstractmethod
    # def visualize(self):
    #     """
    #     Visualize trajectories and value functions.
    #     """
    #     raise NotImplementedError

    def step(self, action):
        """
        Gym style step function. Apply action, move robot, get observation,
        calculate reward, check if done.
        
        Assume action in [x,y,yaw,grasp]
        """
        delta_x, delta_y, delta_yaw, grasp = action
        ee_pos, ee_quat = self.get_ee()
        ee_pos_nxt = ee_pos
        ee_pos_nxt[0] += delta_x
        ee_pos_nxt[1] += delta_y
        ee_pos_nxt[2] -= 0.05  #!

        # Move
        ee_quat_nxt = quatMult(euler2quat([delta_yaw, 0., 0.]), ee_quat)
        self.move(ee_pos_nxt, absolute_global_quat=ee_quat_nxt, numSteps=300)

        # Open or close gripper
        if grasp == 1:
            self.grasp(targetVel=0.10)
        elif grasp == -1:
            self.grasp(targetVel=-0.10)
        else:
            raise ValueError

        # Check if all objects removed
        reward = 0
        if grasp == -1:
            self.clear_obj()
            if len(self.obj_id_list) == 0:
                reward = 1
        return self._get_obs(), reward, True, {}

    def _get_obs(self,
                 offset=[0.04, 0, 0.04],
                 img_H=160,
                 img_W=160,
                 camera_fov=90,
                 camera_aspect=1):
        """Wrist camera image
        """
        ee_pos, ee_quat = self.get_ee()
        rot_matrix = quat2rot(ee_quat)
        camera_pos = ee_pos + rot_matrix.dot(offset)
        # plot_frame_pb(camera_pos, ee_orn)

        # rot_matrix = [0, self.camera_tilt / 180 * np.pi, yaw]
        # rot_matrix = self._p.getMatrixFromQuaternion(
        #     self._p.getQuaternionFromEuler(rot_matrix))
        # rot_matrix = np.array(rot_matrix).reshape(3, 3)

        # Initial vectors
        init_camera_vector = (0, 0, 1)  # z-axis
        init_up_vector = (1, 0, 0)  # x-axis

        # Rotated vectors
        camera_vector = rot_matrix.dot(init_camera_vector)
        up_vector = rot_matrix.dot(init_up_vector)
        view_matrix = self._p.computeViewMatrix(
            camera_pos, camera_pos + 0.1 * camera_vector, up_vector)

        # Get Image
        far = 1000.0
        near = 0.01
        projection_matrix = self._p.computeProjectionMatrixFOV(
            fov=camera_fov, aspect=camera_aspect, nearVal=near, farVal=far)
        _, _, rgb, depth, _ = p.getCameraImage(
            img_W,
            img_H,
            view_matrix,
            projection_matrix,
            flags=self._p.ER_NO_SEGMENTATION_MASK)
        depth = far * near / (far - (far - near) * depth)
        return depth
