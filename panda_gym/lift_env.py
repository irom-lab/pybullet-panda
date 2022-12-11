import numpy as np

from .tool import Tool
from util.numeric import unnormalize_tanh
from panda_gym.base_env import BaseEnv
from alano.geometry.transform import quat2euler, euler2quat, quat2rot, quatInverse, quatMult


class LiftEnv(BaseEnv):
    def __init__(
        self,
        task=None,
        render=False,
        camera_param=None,
        #
        mu=0.3,
        sigma=0.01,
    ):
        super(LiftEnv, self).__init__(
            task=task,
            render=render,
            camera_param=camera_param,
        )
        self.obj_id = None 
        self._mu = mu
        self._sigma = sigma

        # Continuous action space
        self._action_low = np.array([-0.1, -0.1, -0.1, 
                                    # -np.pi/4, -np.pi/4, 
                                    -np.pi/4])
        self._action_high = np.array([0.1, 0.1, 0.1, 
                                    #  np.pi/4, np.pi/4, 
                                     np.pi/4])

        # Grasping threshold
        self._grasp_threshold = 0.01
        self._regrasp_threshold = 0.05  # gripper reopens if tip above the threshold and object not grasped
        self._obj_max_dist = 0.2

        # Max EE range
        self._max_ee_x = [0.2, 0.8]
        self._max_ee_y = [-0.5, 0.5]
        self._max_ee_z = 0.6
        self._max_ee_roll = np.pi/6
        self._max_ee_pitch = np.pi/6

        # Target
        self._target_z = 0.1


    @property
    def state_dim(self):
        """
        Dimension of robot state - x, y, z, roll, pitch, yaw, gripper
        """
        return 7


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
        return np.hstack((ee_pos, ee_euler, self._get_gripper_joint()[0:1]))


    def close_pb(self):
        super().close_pb()
        self.obj_id = None


    def reset_task(self, task):
        """
        Reset the task for the environment. Load object - task
        """
        # Clean table
        if self.obj_id is not None:
            self._p.removeBody(self.obj_id)

        # Load tool
        self._tool = Tool(self)
        self.obj_id = self._tool.load(task, mass=0.2)

        # Record object initial pos
        pos, quat = self._tool.get_pose()  # this returns COM, not geometric center!
        self.tool_initial_pos = pos
        self.tool_initial_euler = quat2euler(quat)

        # Record initial dist to target
        self.initial_target_z_dist = np.linalg.norm(pos[2]-self._target_z)

        # Record initial dist to object - use closest point
        self.initial_dist = self._get_min_dist_from_finger(self.obj_id, 
                                                    max_dist=self._obj_max_dist)

        # Flag for grasping
        self.grasp_executed = False


    def reset(self, task=None):
        """
        Reset the environment, including robot state, task, and obstacles.
        Initialize pybullet client if 1st time.
        """
        if task is None:    # use default if not specified
            task = self.task
        self.task = task    # save task
        # task['init_joint_angles'] = joint_poses
        init_x = self.rng.random()*0.04 + 0.48  # [0.48, 0.52]
        init_y = self.rng.random()*0.04 + -0.02 # [-0.02, 0.02]
        init_yaw = self.rng.random()*2*np.pi/3 + -np.pi/3
        self.init_quat = euler2quat([np.pi+init_yaw, np.pi, 0])
        task['init_pose'] = [init_x, init_y, 0.30] + \
                            list(self.init_quat)    # 0.155
        task['initial_finger_vel'] = self._finger_open_vel
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
        target_ang_vel = [0, 0, raw_action[3]]
        # if norm_action[4] > 0:
        #     grasp_vel = norm_action[4]*0.05 + 0.05
        # else:
        #     grasp_vel = norm_action[4]*0.05 - 0.05

        # Apply
        self.move_vel(target_lin_vel, 
                      target_ang_vel, 
                    #   grasp_vel=grasp_vel,
                    #   check_obj_between_finger=True,
                      num_steps=48,
                      apply_grasp_threshold=self._grasp_threshold,
                    #   init_quat=self.init_quat,
                    #   max_roll=self._max_ee_roll,
                    #   max_pitch=self._max_ee_pitch,
                    #   max_roll_vel=self._action_high[3],
                    #   max_pitch_vel=self._action_high[4]
                      )  # 5Hz

        # Check EE
        ee_pos, ee_quat = self._get_ee()

        # Check object
        tool_pos, tool_quat = self._tool.get_pose()
        tool_euler = quat2euler(tool_quat)
        # obj_yaw = quat2euler(obj_quat)[0]
        # obj_initial_yaw = self.obj_initial_euler[0]
        # obj_yaw_rel = min(abs(obj_yaw-obj_initial_yaw), self.max_obj_yaw)
        # yaw_ratio = obj_yaw_rel/self.max_obj_yaw

        # Apply perturbation - no every timestep
        # self._p.applyExternalForce(self.tool_id, -1, forceObj=[0,-10,10], 
        #                         posObj=[0.50, -0.10, 0.1], 
        #                         flags=env._p.WORLD_FRAME)

        # Reward - approaching object
        dist = self._get_min_dist_from_finger(self.obj_id, 
                                             max_dist=self._obj_max_dist)
        dist_radio = dist/self.initial_dist
        reward = max(0, 1-dist_radio)*0.1

        # Reward - Lift
        if self._check_hold_object(self.obj_id):
            z_dist = np.linalg.norm(tool_pos[2] - self._target_z)
            z_dist_ratio = z_dist/self.initial_target_z_dist
            reward += max(0, 1-z_dist_ratio)

            # If object moved
            tool_pos_delta = np.linalg.norm(tool_pos[:2]-self.tool_initial_pos[:2])
            if tool_pos_delta > 0.1:
                reward -= 0.2
            # if abs(tool_euler[0]-self.tool_initial_euler[0]) > 0.785:
            #     reward -= 0.2
            # tool_euler_delta = np.linalg.norm(tool_euler-self.tool_initial_euler)
            # reward -= max(0.1, tool_euler_delta*0.1)

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
        obs_wrist = self.get_wrist_obs(camera_param)  # uint8
        obs_overhead = self.get_overhead_obs(camera_param)  # uint8
        return np.vstack((obs_wrist, obs_overhead))
        # return obs_overhead
