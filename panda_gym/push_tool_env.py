from abc import ABC
import numpy as np
import os
from os.path import dirname

from panda_gym.base_env import BaseEnv, normalize_action
from alano.geometry.transform import quat2euler, euler2quat


class PushToolEnv(BaseEnv, ABC):
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
        super(PushToolEnv, self).__init__(
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
        self.action_low = np.array([-0.05, -0.1, -np.pi/4])
        self.action_high = np.array([0.15, 0.1, np.pi/4])
        self._finger_open_pos = 0.0

        # Max yaw for clipping reward
        self.max_yaw = np.pi/2


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

        # Load urdf
        obj_id = self._p.loadURDF(
            task['obj_path'],
            basePosition=np.array([0.55, 0, 0]+\
                         np.array(task['obj_pos_offset'])),
            baseOrientation=euler2quat(task['obj_euler']), 
            globalScaling=task['obj_scaling']
        )
        self._obj_id_list += [obj_id]

        # Change color
        self._p.changeVisualShape(obj_id, -1,
                                rgbaColor=[1.0, 0.5, 0.3, 1.0])
        for link_ind in range(self._p.getNumJoints(obj_id)):
            self._p.changeVisualShape(obj_id, link_ind,
                                    rgbaColor=[1.0, 0.5, 0.3, 1.0])

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

        # Set target - account for COM offset in y
        # self.target_pos = np.array([0.70, pos[1]])
        self.target_pos = np.array([0.70, 0.10])
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
            plane_urdf_path = os.path.join(dirname(dirname(__file__)), 
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
        self.reset_robot(self._mu, self._sigma, 
                         init_joint_angles=[
                             0, 0.35, 0, -2.813, 0, 3.483, 0.785])
        self.grasp(target_vel=0)

        # Reset task - add object before arm down
        self.reset_task(task)

        # Reset timer
        self.step_elapsed = 0
        
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

        # Check arm pose
        ee_pos, ee_orn = self._get_ee()
        ee_euler = quat2euler(ee_orn)

        # Check reward
        box_pos, box_quat = self._p.getBasePositionAndOrientation(self._obj_id_list[-1])
        box_yaw = min(abs(quat2euler(box_quat)[0]), self.max_yaw)
        dist = np.linalg.norm(box_pos[:2] - self.target_pos)
        yaw_weight = 0. #!
        dist_ratio = dist/self.initial_dist
        yaw_ratio = box_yaw/self.max_yaw
        if dist_ratio < 1.0:
        # if dist_ratio < 1.0 and yaw_ratio < 1.0:
            reward = (1-dist_ratio)*(1-yaw_weight) + (1-yaw_ratio)*yaw_weight
        else:
            reward = 0

        # Check done - terminate early if ee out of bound, do not terminate even reaching the target
        done = False
        if abs(ee_pos[0] - 0.5) > 0.3 or abs(ee_pos[1]) > 0.3:
            done = True

        # Return info
        info = {}
        info['task'] = self.task
        info['ee_pos'] = ee_pos
        info['ee_orn'] = ee_orn
        info['success'] = False
        if reward > 0.9:
            info['success'] = True
        return self._get_obs(self._camera_params), reward, done, info


    def _get_obs(self, camera_params):
        obs = self.get_overhead_obs(camera_params)  # uint8
        return obs
