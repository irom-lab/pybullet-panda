from abc import ABC
import numpy as np

from panda_gym.base_env import normalize_action
from panda_gym.grasp_env import GraspEnv
from alano.geometry.transform import quatMult, euler2quat, euler2quat

class GraspMultiViewEnv(GraspEnv, ABC):
    def __init__(
        self,
        task=None,
        renders=False,
        use_rgb=False,
        use_depth=True,
        #
        mu=0.5,
        sigma=0.03,
        camera_params=None,
    ):
        """
        Args:
            task (str, optional): the name of the task. Defaults to None.
            img_H (int, optional): the height of the image. Defaults to 128.
            img_W (int, optional): the width of the image. Defaults to 128.
            use_rgb (bool, optional): whether to use RGB image. Defaults to
                True.
            renders (bool, optional): whether to render the environment.
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
            use_rgb=use_rgb,
            use_depth=use_depth,
            mu=mu,
            sigma=sigma,
            camera_params=camera_params,  #TODO
        )

        # Wrist view camera
        self.camera_fov = 50    # https://www.intel.com/content/www/us/en/support/articles/000030385/emerging-technologies/intel-realsense-technology.html
        self.camera_aspect = 1.5
        self.camera_wrist_offset = [0.03, -0.0325, 0.06]  # https://support.intelrealsense.com/hc/en-us/community/posts/360047245353-Locating-the-origin-of-the-reference-system
        self.camera_max_depth = 1

        # Continuous action space
        self.action_low = np.array(
                [-0.02, -0.02, -0.05, -15 * np.pi / 180])
        self.action_high = np.array(
                [0.02, 0.02, -0.03, 15 * np.pi / 180])  #! TODO: tune

        # Fix seed
        self.seed(0)


    @property
    def action_dim(self):
        """
        Dimension of robot action - x,y,yaw
        """
        return 4


    def reset_task(self, task):
        """
        Reset the task for the environment. Load object - task
        """
        # Clean table
        for obj_id in self._obj_id_list:
            self._p.removeBody(obj_id)

        # Reset obj info
        self._obj_id_list = []
        self._obj_initial_pos_list = {}

        # Load all
        obj_path_list = task['obj_path_list']
        obj_state_all = task['obj_state_all']
        obj_rgba_all = task['obj_rgba_all']
        obj_scale_all = task['obj_scale_all']
        obj_height_all = task['obj_height_all']
        for obj_path, obj_state, obj_rgba, obj_scale, obj_height in zip(obj_path_list, obj_state_all, obj_rgba_all, obj_scale_all, obj_height_all):
            # obj_state[3:] = 0
            # obj_path = '/home/allen/data/processed_objects/SNC_v4_mug/0.urdf'
            # obj_path = '/home/allen/data/processed_objects/SNC_v4/03797390/16.urdf'

            # Use mug height for initial z
            obj_state[2] = obj_height / 2 + 0.005

            # load urdf
            obj_id = self._p.loadURDF(
                obj_path,
                basePosition=obj_state[:3],
                baseOrientation=euler2quat(obj_state[3:]), 
                globalScaling=obj_scale)
            self._obj_id_list += [obj_id]

            # Infer number of links - change dynamics for each
            num_joint = self._p.getNumJoints(obj_id)
            link_all = [-1] + [*range(num_joint)]
            for link_id in link_all:
                self._p.changeDynamics(
                    obj_id,
                    link_id,
                    lateralFriction=self._mu,
                    spinningFriction=self._sigma,
                    frictionAnchor=1,
                )

            # Change color - assume base link
            # texture_id = self._p.loadTexture('/home/allen/data/dtd/images/banded/banded_0002_t.jpg')
            for link_id in link_all:
                self._p.changeVisualShape(obj_id, link_id,
                                        #   textureUniqueId=texture_id) 
                                          rgbaColor=obj_rgba)

            # Let objects settle (actually do not need since we know the height of object and can make sure it spawns very close to table level)
            for _ in range(50):
                self._p.stepSimulation()

        # Record object initial height (for comparing with final height when checking if lifted). Note that obj_initial_height_list is a dict
        for obj_id in self._obj_id_list:
            pos, _ = self._p.getBasePositionAndOrientation(obj_id)
            self._obj_initial_pos_list[obj_id] = pos


    def step(self, action):
        """
        Gym style step function. Apply action, move robot, get observation,
        calculate reward, check if done.
        
        Assume action in [x,y,yaw,grasp]
        """
        # Count time
        self.step_elapsed += 1

        # Extract action
        norm_action = normalize_action(action, self.action_low, self.action_high)
        delta_x, delta_y, delta_z, delta_yaw = norm_action
        ee_pos, ee_quat = self._get_ee()
        ee_pos_nxt = ee_pos
        ee_pos_nxt[0] += delta_x
        ee_pos_nxt[1] += delta_y
        ee_pos_nxt[2] += delta_z

        # Move
        ee_quat[2:] = 0 # correct roll and pitch in case of collision in last step
        ee_quat_nxt = quatMult(
            euler2quat([delta_yaw, 0, 0]), ee_quat)
        collision_force_threshold = 0
        if self.step_elapsed == self.max_steps: # only check collision at last step
            collision_force_threshold = 20
        collision = self.move_pose(absolute_pos=ee_pos_nxt,
                            absolute_global_quat=ee_quat_nxt,
                            num_steps=100,
                            collision_force_threshold=collision_force_threshold,
                            )

        # Grasp if last step, and then lift
        if self.step_elapsed == self.max_steps:
            self.grasp(target_vel=-0.10)  # close
            self.move_pose(ee_pos_nxt,  # keep for some time
                            absolute_global_quat=ee_quat_nxt,
                            num_steps=100)
            ee_pos_nxt[2] += 0.1
            self.move_pose(ee_pos_nxt,
                            absolute_global_quat=ee_quat_nxt,
                            num_steps=100)
        else:
            self.grasp(target_vel=0.10)  # open

        # Check if all objects removed
        reward = 0.2
        success = 0
        if self.step_elapsed == self.max_steps:
            num_obj_clear = self.clear_obj(thres=0.03)
            if num_obj_clear > 0:
                reward = 1
                success = 1
        if collision: # only check at last step
            reward = 0
            self.safe_trial = False

        # Check termination condition
        done = False
        if self.step_elapsed == self.max_steps:
            done = True
        return self._get_obs(), reward, done, {'success': success, 'safe': self.safe_trial}


    def _get_obs(self):
        out = self.get_wrist_obs()
        return out

        # import pkgutil
        # egl = pkgutil.get_loader('eglRenderer')
        # import pybullet_data
        # self._p.setAdditionalSearchPath(pybullet_data.getDataPath())
        # plugin = self._p.loadPlugin(egl.get_filename(), "_eglRendererPlugin")
        # print("plugin=", plugin)

        # from PIL import Image
        # import matplotlib.pyplot as plt
        # f, axarr = plt.subplots(1, 2)
        # axarr[0].imshow(out[0])
        # axarr[1].imshow(np.transpose(out[1:], (1, 2, 0)))
        # plt.show()
        # im_rgb = Image.fromarray(np.transpose(out[1:], (1, 2, 0)))
        # im_rgb.save('rgb_sample_0.jpg')
        # im_depth = Image.fromarray(out[0])
        # im_depth.save('depth_sample_0.jpg')

        # Check if object moved or gripper rolled or pitched, meaning contact happened
        # flag_obj_move = False
        # flag_ee_move = False
        # for obj_id in self._obj_id_list:
        #     pos, _ = self._p.getBasePositionAndOrientation(obj_id)
        #     if math.sqrt((pos[0] - self._obj_initial_pos_list[obj_id][0])**2 + (pos[1] - self._obj_initial_pos_list[obj_id][1])**2) > 0.01:
        #         # print('moved ', obj_id)
        #         flag_obj_move = True
        #         break
        # ee_euler = quat2euler(self._get_ee()[1])
        # if abs(ee_euler[1]) > 15 * np.pi / 180 or abs(ee_euler[2]) < (180 - 15) * np.pi / 180:
        #     flag_ee_move = True
            # print('ee moved')
