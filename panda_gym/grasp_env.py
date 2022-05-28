import numpy as np
import os
from os.path import dirname

from panda_gym.base_env import BaseEnv
from alano.geometry.transform import quatMult, euler2quat


class GraspEnv(BaseEnv):
    def __init__(
        self,
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
            render (bool, optional): whether to render the environment.
                Defaults to False.
            max_steps_train (int, optional): the maximum number of steps to
                train. Defaults to 100.
            max_steps_eval (int, optional): the maximum number of steps to
                evaluate. Defaults to 100.
            done_type (str, optional): the type of the done. Defaults to
                'fail'.
        """
        super(GraspEnv, self).__init__(
            renders=renders,
            img_h=img_h,
            img_w=img_w,
            use_rgb=use_rgb,
            use_depth=use_depth,
            max_steps_train=max_steps_train,
            max_steps_eval=max_steps_eval,
            done_type=done_type,
        )
        self._mu = mu
        self._sigma = sigma

        # Object id
        self._obj_id_list = []
        self._obj_initial_pos_list = {}

        # Camera info
        self._camera_params = camera_params


    @property
    def state_dim(self):
        """
        Dimension of robot state - 3D + gripper
        """
        return 4


    @property
    def action_dim(self):
        """
        Dimension of robot action - x, y, yaw
        """
        return 3


    @property
    def init_joint_angles(self):
        """
        Initial joint angles for the task - [0.45, 0, 0.40], straight down - ee to finger tip is 15.5cm
        """
        return [
            0, -0.255, 0, -2.437, 0, 2.181, 0.785, 0, 0,
            self._finger_open_pos, 0.00, self._finger_open_pos, 0.00
        ]


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

        # Load urdf
        obj_path = 'data/sample/mug/3.urdf'
        obj_id = self._p.loadURDF(
            obj_path,
            basePosition=[0.5, 0.0, 0.15],
            baseOrientation=[0, 0, 0, 1])
        self._obj_id_list += [obj_id]

        # Let objects settle (actually do not need since we know the height of object and can make sure it spawns very close to table level)
        for _ in range(50):
            self._p.stepSimulation()

        # Record object initial height (for comparing with final height when checking if lifted). Note that obj_initial_height_list is a dict
        for obj_id in self._obj_id_list:
            pos, _ = self._p.getBasePositionAndOrientation(obj_id)
            self._obj_initial_pos_list[obj_id] = pos


    def reset(self, task=None):
        """
        Reset the environment, including robot state, task, and obstacles.
        Initialize pybullet client if 1st time.
        """
        if self._physics_client_id < 0:

            # Initialize PyBullet instance
            self.init_pb()

            # Load table
            plane_urdf_path =  os.path.join(dirname(dirname(__file__)), 
                                            f'data/plane/plane.urdf')
            self._plane_id = self._p.loadURDF(plane_urdf_path,
                                              basePosition=[0, 0, -1],
                                              useFixedBase=1)
            table_urdf_path =  os.path.join(dirname(dirname(__file__)),
                                            f'data/table/table.urdf')
            self._table_id = self._p.loadURDF(
                table_urdf_path,
                basePosition=[0.400, 0.000, -0.630 + 0.005],
                baseOrientation=[0., 0., 0., 1.0],
                useFixedBase=1)

            # Set friction coefficient for table
            self._p.changeDynamics(
                self._table_id,
                -1,
                lateralFriction=self._mu,
                spinningFriction=self._sigma,
                frictionAnchor=1,
            )

            # Change color
            self._p.changeVisualShape(self._table_id, -1,
                                    rgbaColor=[0.7, 0.7, 0.7, 1.0])

            # Reset object info
            self._obj_id_list = []
            self._obj_initial_pos_list = {}

        # Reset task - add object before arm down
        if self._panda_id >= 0:
            self.reset_robot_joints(self.up_joint_angles)
        self.reset_task(task)

        # Load arm, no need to settle (joint angle set instantly)
        self.reset_robot(self._mu, self._sigma)

        # Reset timer
        self.step_elapsed = 0
        
        # Reset safety of the trial
        self.safe_trial = True

        return self._get_obs(self._camera_params)


    def step(self, action):
        """
        Gym style step function. Apply action, move robot, get observation,
        calculate reward, check if done.
        
        Assume action in [x,y,yaw]
        """

        # Set arm to starting pose
        initial_ee_pos_before_img = np.array([0.3, -0.5, 0.25])
        initial_ee_orn = np.array([1.0, 0.0, 0.0, 0.0])  # straight down
        self.reset_arm_joints_ik(initial_ee_pos_before_img, initial_ee_orn)
        self.grasp(target_vel=0.10)  # open gripper

        # Execute, reset ik on top of object, reach down, grasp, lift, check success
        ee_pos = action[:3]
        ee_pos_before = ee_pos + np.array([0, 0, 0.10])
        ee_pos_after = ee_pos + np.array([0, 0, 0.05])
        ee_orn = quatMult(euler2quat([action[-1], 0., 0.]), initial_ee_orn)
        for _ in range(3):
            self.reset_arm_joints_ik(ee_pos_before, ee_orn)
            self._p.stepSimulation()
        self.move_pose(ee_pos, absolute_global_quat=ee_orn, num_steps=300)
        self.grasp(target_vel=-0.10)  # always close gripper
        self.move_pose(ee_pos, absolute_global_quat=ee_orn,
                  num_steps=100)  # keep pose until gripper closes
        self.move_pose(ee_pos_after, absolute_global_quat=ee_orn,
                  num_steps=150)  # lift

        # Check if all objects removed
        self.clear_obj()
        if len(self._obj_id_list) == 0:
            reward = 1
        else:
            reward = 0
        return self._get_obs(self._camera_params), reward, True, {}


    def clear_obj(self, thres=0.03):
        height = []
        obj_to_be_removed = []
        for obj_id in self._obj_id_list:
            pos, _ = self._p.getBasePositionAndOrientation(obj_id)
            height += [pos[2]]
            if pos[2] - self._obj_initial_pos_list[obj_id][2] > thres:
                obj_to_be_removed += [obj_id]

        for obj_id in obj_to_be_removed:
            self._p.removeBody(obj_id)
            self._obj_id_list.remove(obj_id)
        return len(obj_to_be_removed)


    def _get_obs(self, camera_params):
        return self.get_overhead_obs(camera_params)
