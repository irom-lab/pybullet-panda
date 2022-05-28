import numpy as np

from .tool import Tool
from .util import normalize_action
from panda_gym.push_env import PushEnv
from alano.geometry.transform import quat2euler


class PushToolEnv(PushEnv):
    def __init__(
        self,
        task=None,
        renders=False,
        use_rgb=True,
        use_depth=False,
        #
        mu=0.5, #!
        sigma=0.1,
        camera_params=None,
    ):
        super(PushToolEnv, self).__init__(
            task=task,
            renders=renders,
            use_rgb=use_rgb,
            use_depth=use_depth,
            mu=mu,
            sigma=sigma,
            camera_params=camera_params,
        )
        self.target_pos = np.array([0.75, 0.15])    # TODO: add to task


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
        obj_id = self._tool.load(task)
        self.obj_id = obj_id

        # Record object initial pos
        pos, quat = self._tool.get_pose()  # this returns COM, not geometric center!
        self._tool_initial_pos = pos
        self._tool_initial_euler = quat2euler(quat)

        # Set target - account for COM offset in y
        self.initial_dist = np.linalg.norm(pos[:2] - self.target_pos)


    def reset(self, task=None):
        """
        Reset the environment, including robot state, task, and obstacles.
        Initialize pybullet client if 1st time.
        """
        if task is None:    # use default if not specified
            task = self.task
        self.task = task    # save task
        task['init_joint_angles'] = [0, 0.35, 0, -2.813, 0, 3.483, 0.785]
        task['init_joint_angles'] += [0, 0, self._finger_close_pos, 0.0,
                                    self._finger_close_pos, 0.0]
        task['initial_finger_vel'] = self._finger_close_vel  # keep finger closed
        return super().reset(task)


    def step(self, action):
        """
        Gym style step function. Apply action, move robot, get observation,
        calculate reward, check if done.
        
        Assume action in [x,y, yaw] velocity. Right now velocity control is instantaneous, not accounting for acceleration
        """
        # Keep gripper closed
        self.grasp(self._finger_close_vel)

        # Apply action - velocity control
        norm_action = normalize_action(action, self._action_low, 
                                            self._action_high)
        x_vel, y_vel, yaw_vel = norm_action
        target_lin_vel = [x_vel, y_vel, 0]
        target_ang_vel = [0, 0, yaw_vel]
        self.move_vel(target_lin_vel, target_ang_vel, num_steps=48) # 5Hz

        # Check EE
        ee_pos, ee_orn = self._get_ee()
        ee_euler = quat2euler(ee_orn)

        # Check object
        tool_pos, tool_quat = self._tool.get_pose()
        tool_yaw = quat2euler(tool_quat)[0]
        tool_initial_yaw = self._tool_initial_euler[0]
        tool_yaw_rel = min(abs(tool_yaw-tool_initial_yaw), self._max_obj_yaw)
        # yaw_ratio = obj_yaw_rel/self._max_obj_yaw

        # Reward - [0,1]
        dist = np.linalg.norm(tool_pos[:2] - self.target_pos)
        dist_ratio = dist/self.initial_dist
        reward = max(0, 1-dist_ratio)

        # Check done - terminate early if ee out of bound, do not terminate even reaching the target
        done = False
        if ee_pos[0] < self._max_ee_x[0] or ee_pos[0] > self._max_ee_x[1] \
            or ee_pos[1] < self._max_ee_y[0] or ee_pos[1] > self._max_ee_y[1]:
            done = True

        # Return info
        info = {}
        info['task'] = self.task
        info['s'] = self.state
        return self._get_obs(self._camera_params), reward, done, info
