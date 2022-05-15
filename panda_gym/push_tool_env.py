import numpy as np

from panda_gym.push_env import PushEnv, normalize_action
from alano.geometry.transform import quat2euler, euler2quat


class PushToolEnv(PushEnv):
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
            mu=mu,
            sigma=sigma,
            camera_params=camera_params,
        )
        self._tool_color = [1.0, 0.5, 0.3, 1.0]


    def reset_task(self, task):
        """
        Reset the task for the environment. Load object - task
        """
        # Clean table
        for obj_id in self._obj_id_list:
            self._p.removeBody(obj_id)

        # Reset obj info
        self._obj_id_list = []
        self._obj_initial_pos_list = []
        self._obj_initial_euler_list = []

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
        self._p.changeVisualShape(obj_id, -1, rgbaColor=self._tool_color)
        for link_ind in range(self._p.getNumJoints(obj_id)):
            self._p.changeVisualShape(obj_id, link_ind, 
                                      rgbaColor=self._tool_color)

        # Let objects settle (actually do not need since we know the height of object and can make sure it spawns very close to table level)
        for _ in range(50):
            # Send velocity commands to joints
            for i in range(self._num_joint_arm):
                self._p.setJointMotorControl2(self._panda_id,
                    i,
                    self._p.VELOCITY_CONTROL,
                    targetVelocity=0,
                    force=self._max_joint_force[i],
                    maxVelocity=self._joint_max_vel[i],
                )
            self._p.stepSimulation()

        # Record object initial pos
        for obj_id in self._obj_id_list:
            pos, quat = self._p.getBasePositionAndOrientation(obj_id)  # this returns COM, not geometric center!
            self._obj_initial_pos_list += [pos]
            self._obj_initial_euler_list += [quat2euler(quat)]

        # Set target - account for COM offset in y
        self.target_pos = np.array([0.70, 0.10])    # TODO: add to task
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
        return super().reset(task)


    def step(self, action):
        """
        Gym style step function. Apply action, move robot, get observation,
        calculate reward, check if done.
        
        Assume action in [x,y, yaw] velocity. Right now velocity control is instantaneous, not accounting for acceleration
        """

        # Apply action - velocity control
        norm_action = normalize_action(action, self.action_low, self.action_high)
        x_vel, y_vel, yaw_vel = norm_action
        target_lin_vel = [x_vel, y_vel, 0]
        target_ang_vel = [0, 0, yaw_vel]
        self.move_vel(target_lin_vel, target_ang_vel, num_steps=48) # 5Hz

        # Check EE
        ee_pos, ee_orn = self._get_ee()
        ee_euler = quat2euler(ee_orn)

        # Check object
        obj_pos, obj_quat = self._p.getBasePositionAndOrientation(self._obj_id_list[-1])
        obj_yaw = quat2euler(obj_quat)[0]
        obj_initial_yaw = self._obj_initial_euler_list[0][0] # only one object
        obj_yaw_rel = min(abs(obj_yaw-obj_initial_yaw), self.max_obj_yaw)
        yaw_ratio = obj_yaw_rel/self.max_obj_yaw
        print(obj_yaw_rel, obj_yaw)

        # Reward - [0,1]
        dist = np.linalg.norm(obj_pos[:2] - self.target_pos)
        dist_ratio = dist/self.initial_dist
        reward = max(0, 1-dist_ratio)

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
