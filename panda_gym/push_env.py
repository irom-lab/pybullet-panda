from abc import ABC
import numpy as np
import os
from os.path import dirname

from panda_gym.base_env import BaseEnv, normalize_action
from alano.geometry.transform import quat2euler, euler2quat
from alano.bullet.kinematics import full_jacob_pb


class PushEnv(BaseEnv, ABC):
    def __init__(
        self,
        task=None,
        renders=False,
        use_rgb=False,
        use_depth=True,
        #
        mu=0.3,
        sigma=0.01,
        camera_params=None,
    ):
        """
        Args:
            task (str, optional): the name of the task. Defaults to None.
            use_rgb (bool, optional): whether to use RGB image. Defaults to
                True.
            render (bool, optional): whether to render the environment.
                Defaults to False.
        """
        super(PushEnv, self).__init__(
            task=task,
            renders=renders,
            use_rgb=use_rgb,
            use_depth=use_depth,
            camera_params=camera_params,
        )
        self._mu = mu
        self._sigma = sigma

        # Object id
        self._obj_id_list = []
        self._obj_initial_pos_list = {}

        # Continuous action space
        # self.action_low = np.array([-0.1, -0.3, -np.pi/4])
        # self.action_high = np.array([0.3, 0.3, np.pi/4])
        self.action_low = np.array([-0.05, -0.1, -np.pi/4])
        self.action_high = np.array([0.15, 0.1, np.pi/4])
        self._finger_open_pos = 0.0

        # Max object range
        self.max_obj_yaw = np.pi/2

        # Max EE range
        self.max_ee_x = [0.2, 0.8]
        self.max_ee_y = [-0.3, 0.3]


    @property
    def action_dim(self):
        """
        Dimension of robot action - x,y,yaw
        """
        return 3


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

        # Reset robot
        # self.reset_arm_joints_ik([0.40, task['ee_y'], 0.18], euler2quat([0, 5*np.pi/6, 0]), gripper_closed=True)

        # Load urdf
        box_collision_id = self._p.createCollisionShape(
            self._p.GEOM_BOX, halfExtents=task['obj_half_dim']
        )
        box_visual_id = self._p.createVisualShape(
            self._p.GEOM_BOX, rgbaColor=[0.3,0.4,0.1,1.0], 
            halfExtents=task['obj_half_dim']
        )
        obj_id = self._p.createMultiBody(
            baseMass=task['obj_mass'],
            baseCollisionShapeIndex=box_collision_id,
            baseVisualShapeIndex=box_visual_id,
            basePosition=task['obj_pos'],
            baseOrientation=self._p.getQuaternionFromEuler([0,0,task['obj_yaw']]),
            baseInertialFramePosition=task['obj_com_offset'],
        )
        self._obj_id_list += [obj_id]

        # Set target - account for COM offset
        self.target_pos = np.array([0.70, task['obj_com_offset'][1]])

        # Let objects settle (actually do not need since we know the height of object and can make sure it spawns very close to table level)
        for _ in range(50):
            # Send velocity commands to joints
            for i in range(self._num_joint_arm):
                self._p.setJointMotorControl2(
                    self._panda_id,
                    i,
                    self._p.VELOCITY_CONTROL,
                    targetVelocity=0,
                    force=self._max_joint_force[i],
                    maxVelocity=self._joint_max_vel[i],
                )
            self._p.stepSimulation()

        # Record object initial pos
        for obj_id in self._obj_id_list:
            pos, _ = self._p.getBasePositionAndOrientation(obj_id)  # this returns COM, not geometric center!
            self._obj_initial_pos_list[obj_id] = pos
        self.initial_dist = np.linalg.norm(pos[:2] - self.target_pos)


    def reset(self, task=None):
        """
        Reset the environment, including robot state, task, and obstacles.
        Initialize pybullet client if 1st time.
        """
        if task is None:    # use default if not specified
            task = self.task
        self.task = task    # save task

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

        # Load arm, no need to settle (joint angle set instantly)
        self.reset_robot(self._mu, self._sigma, task['init_joint_angles'])
        self.grasp(target_vel=0)

        # Reset task - add object before arm down
        self.reset_task(task)

        return self._get_obs(self._camera_params)


    def step(self, action):
        """
        Gym style step function. Apply action, move robot, get observation,
        calculate reward, check if done.
        
        Assume action in [x,y, yaw] velocity. Right now velocity control is instantaneous, not accounting for acceleration
        """
        # Extract action
        norm_action = normalize_action(action, self.action_low, self.action_high)
        x_vel, y_vel, yaw_vel = norm_action
        target_lin_vel = [x_vel, y_vel, 0]
        target_ang_vel = [0, 0, yaw_vel]
        self.move_vel(target_lin_vel, target_ang_vel, num_steps=48) # 5Hz

        # Check EE
        ee_pos, ee_orn = self._get_ee()
        ee_euler = quat2euler(ee_orn)

        # Check reward
        obj_pos, obj_quat = self._p.getBasePositionAndOrientation(self._obj_id_list[-1])
        obj_yaw = min(abs(quat2euler(obj_quat)[0]), self.max_obj_yaw)
        dist = np.linalg.norm(obj_pos[:2] - self.target_pos)
        yaw_weight = 0.8
        dist_ratio = dist/self.initial_dist
        yaw_ratio = obj_yaw/self.max_obj_yaw
        if dist_ratio < 0.2 and yaw_ratio < 0.2:
            reward = (1-dist_ratio/0.2)*(1-yaw_weight) + (1-yaw_ratio/0.2)*yaw_weight
        else:
            reward = 0

        # Check done - terminate early if ee out of bound, do not terminate even reaching the target
        done = False
        if ee_pos[0] < self.max_ee_x[0] or ee_pos[0] > self.max_ee_x[1] \
            or ee_pos[1] < self.max_ee_y[0] or ee_pos[1] > self.max_ee_y[1]:
            done = True

        # Return info
        info = {}
        info['task'] = self.task
        info['ee_pos'] = ee_pos
        info['ee_orn'] = ee_orn
        return self._get_obs(self._camera_params), reward, done, info


    def _get_obs(self, camera_params):
        obs = self.get_overhead_obs(camera_params)  # uint8
        return obs


    # @property
    # def init_joint_angles(self):
    #     """
    #     Initial joint angles for the task - [0.45, 0, 0.40], straight down - ee to finger tip is 15.5cm
    #     """
    #     return [
    #         0, 0.277, 0, -2.813, 0, 3.483, 0.785, 0, 0,
    #         self._finger_open_pos, 0.00, self._finger_open_pos, 0.00
    #     ]
