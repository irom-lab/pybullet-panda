from abc import ABC
import numpy as np
import pybullet_data

from panda_gym.base_env import BaseEnv, normalize_action
from alano.geometry.transform import quat2euler
from alano.bullet.kinematics import full_jacob_pb


class PushEnv(BaseEnv, ABC):
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
        self._urdf_root = pybullet_data.getDataPath()

        # Continuous action space
        self.action_low = np.array([-0.05, -0.2, -np.pi/6]) # 0.1
        self.action_high = np.array([0.2, 0.2, np.pi/6]) # 0.1
        self._finger_open_pos = 0.0 #! 0.01
        self.target_pos = np.array([0.70,0.15]) # 0.

    @property
    def action_dim(self):
        """
        Dimension of robot action - x,y,yaw
        """
        return 3

    @property
    def init_joint_angles(self):
        """
        Initial joint angles for the task - [0.45, 0, 0.40], straight down - ee to finger tip is 15.5cm
        """
        return [
            0, 0.277, 0, -2.813, 0, 3.483, 0.785, 0, 0,
            self._finger_open_pos, 0.00, self._finger_open_pos, 0.00
        ]

    def report(self):
        """
        Print information of robot dynamics and observation.
        """
        raise NotImplementedError

    def visualize(self):
        """
        Visualize trajectories and value functions.
        """
        raise NotImplementedError

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
            pos, _ = self._p.getBasePositionAndOrientation(obj_id)
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
            self._plane_id = self._p.loadURDF(self._urdf_root + '/plane.urdf',
                                              basePosition=[0, 0, -1],
                                              useFixedBase=1)
            self._table_id = self._p.loadURDF(
                self._urdf_root + '/table/table.urdf',
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
        self.reset_robot(self._mu, self._sigma)
        self.grasp(target_vel=0)

        # Reset task - add object before arm down
        self.reset_task(task)

        # Reset timer
        self.step_elapsed = 0
        
        # Reset safety of the trial
        self.safe_trial = True

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
        self.move(target_lin_vel, target_ang_vel, num_steps=48) # 5Hz

        # Check arm pose
        ee_pos, ee_orn = self._get_ee()
        ee_euler = quat2euler(ee_orn)

        # Check reward and done (terminate early if ee out of bound, do not terminate even reaching the target)
        box_pos, _ = self._p.getBasePositionAndOrientation(self._obj_id_list[-1])
        dist = np.linalg.norm(box_pos[:2] - self.target_pos)
        reward = (1 - dist/self.initial_dist)
        # if reward < 0.8:    #! sparse, and non-negative
        #     reward = 0
        done = False
        if abs(ee_pos[0] - 0.5) > 0.3 or abs(ee_pos[1]) > 0.3 or abs(ee_euler[0]) > np.pi/2:
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

    def move(self,
                target_lin_vel,
                target_ang_vel,
                num_steps):
        target_vel = np.hstack((target_lin_vel, target_ang_vel))

        for _ in range(num_steps):
            joint_poses = list(
                np.hstack((self._get_arm_joints(), np.array([0, 0]))))  # add fingers
            ee_state = self._p.getLinkState(self._panda_id,
                                     self._ee_link_id,
                                     computeLinkVelocity=1,
                                     computeForwardKinematics=1)

            # Get the Jacobians for the CoM of the end-effector link. Note that in this example com_rot = identity, and we would need to use com_rot.T * com_trn. The localPosition is always defined in terms of the link frame coordinates.
            zero_vec = list(np.zeros_like(joint_poses))
            jac_t, jac_r = self._p.calculateJacobian(
                self._panda_id, self._ee_link_id,
                ee_state[2], joint_poses, zero_vec,
                zero_vec)  # use localInertialFrameOrientation
            jac_sp = full_jacob_pb(
                jac_t, jac_r)[:, :7]  # 6x10 -> 6x7, ignore last three column
            try:
                joint_dot = np.linalg.pinv(jac_sp).dot(target_vel)
            except np.linalg.LinAlgError:
                joint_dot = np.zeros((7, 1))

            #! Do not account for joint limit for now
            # jointDot = cp.Variable(7)
            # prob = cp.Problem(
            #         cp.Minimize(cp.norm2(jac_sp @ jointDot - target_vel)), \
            #         [jointDot >= -self._panda.jointMaxVel, \
            #         jointDot <= self._panda.jointMaxVel]
            #         )
            # prob.solve()
            # jointDot = jointDot.value

            # Send velocity commands to joints
            for i in range(self._num_joint_arm):
                self._p.setJointMotorControl2(
                    self._panda_id,
                    i,
                    self._p.VELOCITY_CONTROL,
                    targetVelocity=joint_dot[i],
                    force=self._max_joint_force[i],
                    maxVelocity=self._joint_max_vel[i],
                )

            # Keep gripper current velocity
            for id in [self._left_finger_joint_id, self._right_finger_joint_id]:
                self._p.setJointMotorControl2(self._panda_id,
                                        id,
                                        self._p.VELOCITY_CONTROL,
                                        targetVelocity=self._finger_cur_vel,
                                        force=self._max_finger_force,
                                        maxVelocity=0.10)

            # Step simulation, takes 1.5ms
            self._p.stepSimulation()
            # print(
            #     p.getLinkState(self._pandaId,
            #                    self._panda.pandaEndEffectorLinkIndex,
            #                    computeLinkVelocity=1)[6])
            # print(p.getBaseVelocity(objId)[1])
            # print("===")
        return 1
