import numpy as np
import os
from os.path import dirname

from .tool import Tool
from util.numeric import unnormalize_tanh
from util.misc import suppress_stdout
from panda_gym.base_env import BaseEnv
from util.geom import quat2euler, euler2quat


class HammerEnv(BaseEnv):
    def __init__(
        self,
        task=None,
        render=False,
        camera_param=None,
        #
        mu=0.5, #!
        sigma=0.05,
    ):
        super(HammerEnv, self).__init__(
            task=task,
            render=render,
            camera_param=camera_param,
        )
        self.obj_id = None 
        self.block_id = None
        self.peg_id = None
        self._mu = mu
        self._sigma = sigma

        # Continuous action space
        self._action_low = np.array([-0.2, -0.2, -0.2, -np.pi/4])   #!
        self._action_high = np.array([0.2, 0.2, 0.2, np.pi/4])
        self._max_finger_force = 100    #!
        self._finger_close_vel = -0.50

        # Grasping threshold
        self._grasp_threshold = 0.008
        self._regrasp_threshold = 0.05  # gripper reopens if tip above the threshold and object not grasped
        self._obj_max_dist = 0.2
        self._lift_target = 0.05 # should be the same height as peg
        
        # Peg
        self._peg_max_dist = 0.3
        self._peg_contact_margin = 0.002 # if panda and peg within margin, no reward
        self._peg_max_depth = 0.13

        # Max EE range
        self._max_ee_x = [0.2, 0.8]
        self._max_ee_y = [-0.5, 0.5]
        self._max_ee_z = 0.6
        self._max_ee_roll = np.pi/6
        self._max_ee_pitch = np.pi/6


    @property
    def state_dim(self):
        """
        Dimension of robot state - x, y, z, roll, pitch, yaw, gripper
        """
        return 8


    @property
    def action_dim(self):
        """
        Dimension of robot action - x, y, z, yaw
        """
        return 4


    @property
    def state(self):
        ee_pos, ee_orn = self._get_ee()
        ee_euler = quat2euler(ee_orn)
        peg_depth = self.get_peg_depth()
        peg_depth_ratio = (peg_depth + self._peg_max_depth) / (self.initial_peg_depth + self._peg_max_depth)
        return np.hstack((ee_pos, ee_euler, self._get_gripper_joint()[0:1], peg_depth_ratio))


    def close_pb(self):
        super().close_pb()
        self.obj_id = None
        self.block_id = None
        self.peg_id = None


    def reset_task(self, task):
        """
        Reset the task for the environment. Load object - task
        """
        # Clean table
        if self.obj_id is not None:   # todo: do not load again if same tool?
            self._p.removeBody(self.obj_id)

        ############ Tool ############
        self._tool = Tool(self)
        self.obj_id = self._tool.load(task)

        # Record object initial pos
        pos, quat = self._tool.get_pose()  # this returns COM, not geometric center!
        self.tool_initial_pos = pos
        self.tool_initial_euler = quat2euler(quat)

        # Flag for grasping
        self.grasp_executed = False

        # Record initial dist to object - use closest point
        self.initial_dist = self._get_min_dist_from_finger(self.obj_id, 
                                                    max_dist=self._obj_max_dist)

        ############ Peg and block ############

        # More stable peg #?
        # self._p.setPhysicsEngineParameter(numSolverIterations=150, 
        #                                   enableConeFriction=1, 
        #                                   contactBreakingThreshold=1e-4)

        if self.block_id is None:
            block_urdf_path =  os.path.join(dirname(dirname(__file__)), 
                                            f'data/peg_block/peg_block.urdf')
            self.block_id = self._p.loadURDF(block_urdf_path, 
                                        basePosition=[0.50, -0.10, 0], 
                                        baseOrientation=[0, 0, 0.707, 0.707],
                                        useFixedBase=1,
                                        )

        if self.peg_id is None:
            peg_urdf_path = os.path.join(dirname(dirname(__file__)), 
                                            # f'data/peg/peg_new.urdf')
                                            f'data/peg/peg_prim_new.urdf')
            with suppress_stdout():
                self.peg_id = self._p.loadURDF(peg_urdf_path, 
                                            basePosition=[0.50, -0.11, 0.06], 
                                            # baseOrientation=[0, 0, 0.707, 0.707],
                                            baseOrientation=self._p.getQuaternionFromEuler([1.57,0,0]),
                                            # globalScaling=1.0,
                                            flags=self._p.URDF_MERGE_FIXED_LINKS,
                                            )

            # Change color
            self._p.changeVisualShape(self.block_id, -1,    # wood
                                    rgbaColor=[0.8, 0.6, 0.4, 1.0])
            self._p.changeVisualShape(self.peg_id, -1,
                                    rgbaColor=[0.6, 0.6, 0.6, 1.0])
            self._p.changeVisualShape(self.peg_id, 0,
                                    rgbaColor=[0.6, 0.6, 0.6, 1.0])

            # Add resistance to peg
            self._p.changeDynamics(self.peg_id, -1,
                                    lateralFriction=5.0,
                                    spinningFriction=1.0,
                                    frictionAnchor=1,
                                    # collisionMargin=0.0001
                                )
            self._p.changeDynamics(self.peg_id, 0,
                                    lateralFriction=5.0,
                                    spinningFriction=1.0,
                                    frictionAnchor=1,
                                    # collisionMargin=0.0001
                                )
            self._p.changeDynamics(self.block_id, -1,
                                    lateralFriction=5.0,
                                    spinningFriction=1.0,
                                    frictionAnchor=1,
                                )
        else:
            self._p.resetBasePositionAndOrientation(self.peg_id,
                                        posObj=[0.50, -0.11, 0.06], 
                                        ornObj=self._p.getQuaternionFromEuler([1.57,0,0]))

        # Record initial dist to target
        self.initial_lift_dist = np.linalg.norm(pos[2]-self._lift_target)

        # Record initial dist to peg
        self.initial_peg_dist = self._get_min_dist_between_obj(self.obj_id, 
                                                                 self.peg_id, 
                                                    max_dist=self._peg_max_dist)

        # Record initial peg depth - -0.20 if all in
        self.initial_peg_depth = self.get_peg_depth()

        # Flags
        # self._lift_reached = False
        self._peg_reached = False   # no peg depth reward until tool contacted peg


    def get_peg_depth(self):
        return self._p.getBasePositionAndOrientation(self.peg_id)[0][1]


    def get_peg_height(self):
        return self._p.getBasePositionAndOrientation(self.peg_id)[0][2]


    def reset(self, task=None):
        """
        Reset the environment, including robot state, task, and obstacles.
        Initialize pybullet client if 1st time.
        """
        if task is None:    # use default if not specified
            task = self.task
        self.task = task    # save task
        # task['init_joint_angles'] = joint_poses
        init_x = self.rng.random()*0.10 + 0.45  # [0.45, 0.55]
        init_y = self.rng.random()*0.05 + 0.15  # [0.15, 0.20]
        init_z = self.rng.random()*0.05 + 0.30  # [0.30, 0.35]
        init_yaw = self.rng.random()*2*np.pi/2 + -np.pi/2
        self.init_quat = euler2quat([np.pi+init_yaw, np.pi, 0])
        task['init_pose'] = [init_x, init_y, init_z] + \
                            list(self.init_quat)    # 0.155
        task['initial_finger_vel'] = self._finger_open_vel  # keep finger open
        return super().reset(task)


    def step(self, action):
        """
        Gym style step function. Apply action, move robot, get observation,
        calculate reward, check if done.        
        """
        # Before action - check if allowing re-grasp
        if self.grasp_executed and self._get_lowerest_pos()[2] > self._regrasp_threshold and not self._check_hold_object(self.obj_id):
            self.grasp_executed = False

        # Apply action - velocity control
        raw_action = unnormalize_tanh(action, self._action_low, 
                                              self._action_high)
        target_lin_vel = raw_action[:3]
        target_ang_vel = [0, 0, raw_action[-1]]

        # Apply
        self.move_vel(target_lin_vel, 
                      target_ang_vel, 
                      num_steps=48,
                      apply_grasp_threshold=self._grasp_threshold,
                      )  # 5Hz

        # Check EE
        ee_pos, ee_quat = self._get_ee()

        # Check tool
        tool_pos, tool_quat = self._tool.get_pose()

        # Reward - approaching object
        dist = self._get_min_dist_from_finger(self.obj_id, 
                                             max_dist=self._obj_max_dist)
        dist_radio = dist/self.initial_dist
        reward = max(0, 1-dist_radio)*0.1

        if self._check_hold_object(self.obj_id):

            # Reward - approaching peg after grasping
            peg_dist = self._get_min_dist_between_obj(self.obj_id, 
                                                      self.peg_id, 
                                                      max_dist=self._peg_max_dist)
            peg_dist_ratio = peg_dist/self.initial_peg_dist
            reward += max(0, 1-peg_dist_ratio)*0.2

            # Also reward for lifting
            z_dist = np.linalg.norm(tool_pos[2] - self._lift_target)
            z_dist_ratio = z_dist/self.initial_lift_dist
            reward += max(0, 1-z_dist_ratio)*0.5
            
            # Check if tool contacts peg
            # if not self._peg_reached and peg_dist_ratio < 0.2:
            #     self._peg_reached = True

            # Reward - peg moving, and arm not touching peg - block is rigid so okay if arm touches block
            peg_panda_dist = self._get_min_dist_between_obj(self.peg_id, self._panda_id, max_dist=self._peg_contact_margin)
            peg_height = self.get_peg_height()
            # if peg_panda_dist >= self._peg_contact_margin and self._peg_reached and peg_height > 0.04:
            if peg_panda_dist >= self._peg_contact_margin and peg_height > 0.04:
                peg_depth = self.get_peg_depth()
                peg_depth_ratio = (peg_depth + self._peg_max_depth) / (self.initial_peg_depth + self._peg_max_depth)
                if 1-peg_depth_ratio > 0.01:
                    reward += max(0, 1-peg_depth_ratio)*1

            # elif peg_panda_dist < self._peg_contact_margin or peg_height < 0.05:
            #     reward -= 0.05

        # Check done - terminate early if ee out of bound, do not terminate even reaching the target
        done = False
        if ee_pos[0] < self._max_ee_x[0] or ee_pos[0] > self._max_ee_x[1] \
            or ee_pos[1] < self._max_ee_y[0] or ee_pos[1] > self._max_ee_y[1] \
            or ee_pos[2] > self._max_ee_z:
            done = True

        # Return info
        info = {}
        info['task'] = self.task
        info['s'] = self.state
        return self._get_obs(self._camera_param), reward, done, info


    def _get_obs(self, camera_param):
        # obs_wrist = self.get_wrist_obs(camera_param)  # uint8
        obs_overhead = self.get_overhead_obs(camera_param)  # uint8

        camera_param_aux = {}
        camera_param_aux['pos'] = [0.10, 0.40, 0.20]
        camera_param_aux['euler'] = [0, -1.8, 2.5]
        camera_param_aux['img_h'] = 128
        camera_param_aux['img_w'] = 128
        camera_param_aux['aspect'] = 1
        camera_param_aux['fov'] = 70
        camera_param_aux['overhead_min_depth'] = 0.3
        camera_param_aux['overhead_max_depth'] = 0.8

        obs_aux = self.get_overhead_obs(camera_param_aux)  # uint8
        return np.vstack((obs_overhead, obs_aux))

        # return np.vstack((obs_wrist, obs_overhead))
        # return obs_overhead
