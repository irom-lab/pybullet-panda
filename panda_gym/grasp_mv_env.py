from abc import ABC
import numpy as np
import gym

from panda_gym.grasp_env import GraspEnv
from panda.util_geom import quatMult, euler2quat, euler2quat, quat2rot
from alano.geometry.camera import rgba2rgb


class GraspMultiViewEnv(GraspEnv, ABC):
    def __init__(
        self,
        task=None,
        renders=False,
        img_h=128,
        img_w=128,
        use_rgb=False,
        use_depth=True,
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
            renders=renders,
            img_h=img_h,
            img_w=img_w,
            use_rgb=use_rgb,
            use_depth=use_depth,
            max_steps_train=max_steps_train,
            max_steps_eval=max_steps_eval,
            done_type=done_type,
            mu=mu,
            sigma=sigma,
            camera_params=None,  #! use wrist view
        )

        # Wrist view camera
        self.camera_fov = 90
        self.camera_aspect = 1
        self.camera_wrist_offset = [0.04, 0, 0.04]
        self.camera_max_depth = 1

        # Continuous action space
        self.action_lim = np.float32(np.array([
            1.,
            1.,
            1.,
        ]))
        self.action_space = gym.spaces.Box(-self.action_lim, self.action_lim)
        self.action_normalization = [0.02, 0.02, np.pi / 4]

        # Fix seed
        self.seed(0)

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
        self.move(ee_pos_nxt, absolute_global_quat=ee_quat_nxt, num_steps=300)

        # Open or close gripper
        if grasp == 1:
            self.grasp(target_vel=0.10)
        elif grasp == -1:
            self.grasp(target_vel=-0.10)
        else:
            raise ValueError

        # Check if all objects removed
        reward = 0
        if grasp == -1:
            self.clear_obj()
            if len(self.obj_id_list) == 0:
                reward = 1
        return self._get_obs(), reward, True, {}

    def _get_obs(self):
        """Wrist camera image
        """
        ee_pos, ee_quat = self.get_ee()
        rot_matrix = quat2rot(ee_quat)
        camera_pos = ee_pos + rot_matrix.dot(self.camera_wrist_offset)
        # plot_frame_pb(camera_pos, ee_orn)

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
            fov=self.camera_fov,
            aspect=self.camera_aspect,
            nearVal=near,
            farVal=far)
        _, _, rgb, depth, _ = self._p.getCameraImage(
            self.img_w,
            self.img_h,
            view_matrix,
            projection_matrix,
            flags=self._p.ER_NO_SEGMENTATION_MASK)
        out = []
        if self.use_depth:
            depth = far * near / (far - (far - near) * depth)
            depth = (self.camera_max_depth - depth) / self.camera_max_depth
            depth = depth.clip(min=0., max=1.)
            out += [depth[np.newaxis]]
        if self.use_rgb:
            rgb = rgba2rgb(rgb).transpose(2, 0, 1)  # store as uint8
            out += [rgb]
        out = np.concatenate(out)
        print(out.shape)
        return out
