import numpy as np

from .tool import Tool
from util.numeric import unnormalize_tanh
from panda_gym.panda_env import PandaEnv
from util.geom import quat2euler, euler2quat


class SweepEnv(PandaEnv):
    def __init__(
        self,
        task=None,
        render=False,
        camera_param=None,
        #
        mu=0.3,
        sigma=0.01,
    ):
        super(SweepEnv, self).__init__(
            task=task,
            render=render,
            camera_param=camera_param,
        )
        self.tool_id = None
        self.cylinder_id = None
        self._mu = mu
        self._sigma = sigma

        # Continuous action space
        self._action_low = np.array([-0.2, -0.2, -0.2, -np.pi/4])
        self._action_high = np.array([0.2, 0.2, 0.2, np.pi/4])

        # Grasping threshold
        self._grasp_threshold = 0.01
        self._regrasp_threshold = 0.05  # gripper reopens if tip above the threshold and object not grasped
        self._obj_max_dist = 0.2
        self._lift_threshold = 0.05

        # Cylinder
        self._cylinder_max_dist = 0.4
        self._cylinder_contact_margin = 0.02 # if panda and cylinder within margin, no reward
        # self._cylinder_target_x = 0.80
        self._cylinder_target_y = -0.40
        # self._cylinder_init_pos = [0.55, -0.15, 0.1]
        self._cylinder_init_pos = [0.50, 0, 0.1]

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
        return 7


    @property
    def action_dim(self):
        """
        Dimension of robot action - x, y , z, yaw
        """
        return 4


    @property
    def state(self):
        ee_pos, ee_orn = self._get_ee()
        ee_euler = quat2euler(ee_orn)
        return np.hstack((ee_pos, ee_euler, self._get_gripper_joint()[0:1]))


    def close_pb(self):
        super().close_pb()
        self.tool_id = None
        self.cylinder_id = None


    def reset_task(self, task):
        """
        Reset the task for the environment. Load object - task
        """
        # Clean table
        if self.tool_id is not None:   # todo: do not load again if same tool?
            self._p.removeBody(self.tool_id)

        ############ Tool ############
        self._tool = Tool(self)
        self.tool_id = self._tool.load(task)

        # Record object initial pos
        pos, quat = self._tool.get_pose()  # this returns COM, not geometric center!
        self.obj_initial_pos = pos
        self.obj_initial_euler = quat2euler(quat)

        # Flag for grasping
        self.grasp_executed = False

        # Record initial dist to object - use closest point
        self.initial_dist = self._get_min_dist_from_finger(self.tool_id, 
                                                    max_dist=self._obj_max_dist)

        ############ Cylinder to be swept ############

        if self.cylinder_id is None:
            cylinder_collision_id = self._p.createCollisionShape(
                self._p.GEOM_CYLINDER, radius=0.03, height=0.15
            )
            cylinder_visual_id = self._p.createVisualShape(
                self._p.GEOM_CYLINDER, radius=0.03, length=0.15,
                rgbaColor=[0.6, 0.6, 0.6, 1.0], 
            )
            self.cylinder_id = self._p.createMultiBody(
                baseMass=5, #!
                baseCollisionShapeIndex=cylinder_collision_id,
                baseVisualShapeIndex=cylinder_visual_id,
                basePosition=self._cylinder_init_pos,
                baseOrientation=[0, 0, 0, 1],
            )
        else:
            self._p.resetBasePositionAndOrientation(self.cylinder_id,
                                        posObj=self._cylinder_init_pos, 
                                        ornObj=[0, 0, 0, 1])

        # Record initial distance to cylinder
        self.initial_cylinder_dist = self._get_min_dist_between_obj(
                                                            self.tool_id, 
                                                            self.cylinder_id, 
                                            max_dist=self._cylinder_max_dist)

        # Record initial distance to target
        # self.initial_target_dist = np.abs(self._cylinder_target_x - self._cylinder_init_pos[0])
        self.initial_target_dist = np.abs(self._cylinder_target_y - self._cylinder_init_pos[1])


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
        init_y = self.rng.random()*0.10 + 0.10  # [0.10, 0.20]
        init_yaw = self.rng.random()*2*np.pi/2 + -np.pi/2
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
        if self.grasp_executed and self._get_lowerest_pos()[2] > self._regrasp_threshold and not self._check_hold_object(self.tool_id):
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
                    #   init_quat=self.init_quat,
                    #   max_roll=self._max_ee_roll,
                    #   max_pitch=self._max_ee_pitch,
                    #   max_roll_vel=self._action_high[3],
                    #   max_pitch_vel=self._action_high[4]
                      )  # 5Hz

        # Check EE
        ee_pos, ee_quat = self._get_ee()

        # Check tool
        tool_pos, tool_quat = self._tool.get_pose()

        # Check cylinder
        cylinder_pos, _ = self._p.getBasePositionAndOrientation(self.cylinder_id)

        # Reward - approaching tool
        dist = self._get_min_dist_from_finger(self.tool_id, 
                                             max_dist=self._obj_max_dist)
        dist_radio = dist/self.initial_dist
        reward = max(0, 1-dist_radio)*0.1

        if self._check_hold_object(self.tool_id):

            # Reward - lifting
            # if tool_pos[-1] > self._lift_threshold:
            #     reward += 0.2

            # Reward - approaching cylinder after grasping
            cylinder_dist = self._get_min_dist_between_obj(self.tool_id, 
                                                      self.cylinder_id, 
                                                      max_dist=self._cylinder_max_dist)
            cylinder_dist_ratio = cylinder_dist/self.initial_cylinder_dist
            if cylinder_dist_ratio < 0.3:
                reward += 0.1

            # Reward - cylinder moving to target, and arm not touching cylinder
            if self._get_min_dist_between_obj(self.cylinder_id, self._panda_id, max_dist=self._cylinder_contact_margin) >= self._cylinder_contact_margin: 
                # target_dist = np.abs(self._cylinder_target_x - cylinder_pos[0])
                target_dist = np.abs(self._cylinder_target_y - cylinder_pos[1])
                target_dist_ratio = target_dist/self.initial_target_dist
                reward += max(0, 1-target_dist_ratio)
            else:
                reward -= 0.3

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
        camera_param_aux['pos'] = [0.4, 0.6, 0.20]  
        camera_param_aux['euler'] = [0, -1.8, 1.8]
        camera_param_aux['img_h'] = 128
        camera_param_aux['img_w'] = 128
        camera_param_aux['aspect'] = 1
        camera_param_aux['fov'] = 60
        camera_param_aux['overhead_min_depth'] = 0.3
        camera_param_aux['overhead_max_depth'] = 0.8

        obs_aux = self.get_overhead_obs(camera_param_aux)  # uint8
        return np.vstack((obs_overhead, obs_aux))

        # return np.vstack((obs_wrist, obs_overhead))
        # return obs_overhead
