import os
from os.path import dirname
import pathlib
import numpy as np
import torch
import pybullet as p
from pybullet_utils import bullet_client as bc
from collections import deque

from util.image import rgba2rgb
from util.geom import quat2rot, euler2quat, quatMult, log_rot, quatInverse, quat2euler
from util.scaling import traj_time_scaling
from util.bullet import full_jacob_pb, plot_frame_pb
# from util.depth import 


class PandaEnv():
    def __init__(self,
                 task=None,
                 render=False,
                 camera_param=None,
                #  finger_type='drake'
                 ):
        self.task = task
        self.render = render
        if camera_param is None:
            camera_param = {}
            camera_height = 0.40
            camera_param['pos'] = np.array([1.0, 0, camera_height])
            camera_param['euler'] = [0, -3*np.pi/4, 0] # extrinsic - x up, z forward
            camera_param['img_w'] = 64
            camera_param['img_h'] = 64
            camera_param['aspect'] = 1
            camera_param['fov'] = 70    # vertical fov in degrees
            camera_param['max_depth'] = camera_height
        self._camera_param = camera_param

        # PyBullet instance
        self._p = None
        self._physics_client_id = -1  # PyBullet
        self._panda_id = -1

        # Panda config
        # if finger_type is None:
        #     _finger_name = 'panda_arm_finger_orig'
        # elif finger_type == 'long':
        #     _finger_name = 'panda_arm_finger_long'
        # elif finger_type == 'wide_curved':
        #     _finger_name = 'panda_arm_finger_wide_curved'
        # elif finger_type == 'wide_flat':
        #     _finger_name = 'panda_arm_finger_wide_flat'
        # elif finger_type == 'drake':
        #     _finger_name = 'panda_arm_drake'
        #     self._panda_use_inertia_from_file = True
        # else:
        #     raise NotImplementedError
        self._panda_urdf_path = str(pathlib.Path(__file__).parent.parent) + '/data/franka/panda_arm.urdf'
        self._num_joint = 13
        self._num_joint_arm = 7  # Number of joints in arm (not counting hand)
        self._ee_joint_id = 7   # fixed virtual joint
        self._ee_link_id = 8  # hand, index=7 is link8 (virtual one)
        self._left_finger_link_id = 10  # lower
        self._right_finger_link_id = 12
        self._left_finger_joint_id = 9
        self._right_finger_joint_id = 11
        self._max_joint_force = [87, 87, 87, 87, 12, 12, 12]  # from website
        # self._max_finger_force = 20.0
        self._max_finger_force = 35  # office documentation says 70N continuous force
        self._jd = [
            0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001,
            0.00001, 0.00001,
        ]  # joint damping coefficient
        # self._joint_upper_limit = [2.90, 1.76, 2.90, -0.07, 2.90, 3.75, 2.90, 0.04, 0.04]
        # self._joint_lower_limit = [-2.90, -1.76, -2.90, -3.07, -2.90, -0.02, -2.90, -0.04, -0.04]
        # self._joint_range = [5.8, 3.5, 5.8, 3, 5.8, 3.8, 5.8, 0.0, 0.0]
        self._joint_rest_pose = [0, 0.35, 0, -2.813, 0, 3.483, 0.785, 0.0, 0.0]
        self._joint_max_vel = np.array([2.0, 2.0, 2.0, 2.0, 2.5, 2.5, 2.5])  # actually 2.175 and 2.61

        # Initialize current finger pos/vel
        self._finger_cur_pos = 0.04
        self._finger_cur_vel = 0.10
        self._finger_open_pos = 0.04
        self._finger_close_pos = 0.0
        self._finger_open_vel = 0.10
        self._finger_close_vel = -0.10

        # Default joint angles
        self._default_joint_angles = [0, -0.35, 0, -2., 0, 3.483, 0.785, 0.0, 0.0]


    def seed(self, seed=0):
        """Set when vec_env constructed"""
        self.seed_val = seed
        self.rng = np.random.default_rng(seed=self.seed_val)
        torch.manual_seed(self.seed_val)
        torch.cuda.manual_seed(self.seed_val)
        torch.cuda.manual_seed_all(
            self.seed_val)  # if you are using multi-GPU.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


    @property
    def state_dim(self):
        """
        Dimension of robot state
        """
        raise NotImplementedError


    @property
    def action_dim(self):
        """
        Dimension of robot action
        """
        raise NotImplementedError


    @property
    def state(self):
        """
        Current robot state
        """
        raise NotImplementedError


    def step(self):
        """
        Gym style step function. Apply action, move robot, get observation,
        calculate reward, check if done.
        """
        raise NotImplementedError


    def reset_task(self):
        """
        Reset the task for the environment.
        """
        raise NotImplementedError


    def reset(self, task=None):
        """
        Reset the environment, including robot state, task, and obstacles.
        """
        if task is None:
            task = self.task
        self.task = task

        # Initialize if not yet.
        if self._physics_client_id < 0:

            # Initialize PyBullet instance
            self.init_pb()

            # Load table
            plane_urdf_path =  os.path.join(dirname(dirname(__file__)), 
                                            f'data/plane/plane.urdf')
            self._plane_id = self._p.loadURDF(plane_urdf_path,
                                              basePosition=[0, 0, -1],
                                              useFixedBase=1)
            table_urdf_path = os.path.join(dirname(dirname(__file__)),
                                           f'data/table/table.urdf')
            self._table_id = self._p.loadURDF(
                table_urdf_path,
                basePosition=[0.400, 0.000, -0.630 + 0.005],
                baseOrientation=[0., 0., 0., 1.0],
                useFixedBase=1)

            # Set friction coefficient for table
            self._p.changeDynamics(self._table_id, -1,
                                   lateralFriction=self._mu,
                                   spinningFriction=self._sigma,
                                   frictionAnchor=1,
                                   )

            # Change color
            self._p.changeVisualShape(self._table_id, -1,
                                      rgbaColor=[0.7, 0.7, 0.7, 1.0])

        # Load arm - need time for gripper to reset - weird issue with the constraint that the initial finger joint cannot be reset instantly
        self.reset_robot(self._mu, self._sigma, task)
        if hasattr(task, 'initial_finger_vel'):
            init_finger_vel = task['initial_finger_vel']
        else:
            init_finger_vel = self._finger_open_vel # open gripper by defaul
        self.grasp(target_vel=init_finger_vel)
        if self._finger_cur_vel > 0:
            for _ in range(10):
                for i in range(self._num_joint_arm):
                    self._p.setJointMotorControl2(self._panda_id, i,
                                                self._p.VELOCITY_CONTROL,
                                                targetVelocity=0,
                                                force=self._max_joint_force[i],
                                                maxVelocity=0.1)
                self._p.setJointMotorControl2(self._panda_id,
                                            self._left_finger_joint_id,
                                            self._p.VELOCITY_CONTROL,
                                            targetVelocity=self._finger_cur_vel,
                                            force=50,
                                            maxVelocity=1.0)
                self._p.setJointMotorControl2(self._panda_id,
                                            self._right_finger_joint_id,
                                            self._p.VELOCITY_CONTROL,
                                            targetVelocity=self._finger_cur_vel,
                                            force=50,
                                            maxVelocity=1.0)
                self._p.stepSimulation()

        # Reset task - add object before arm down
        self.reset_task(task)

        return self._get_obs(self._camera_param)


    def init_pb(self):
        """
        Initialize PyBullet environment.
        """
        if self.render:
            self._p = bc.BulletClient(connection_mode=p.GUI, options='--width=2000 --height=1600')
            self._p.resetDebugVisualizerCamera(0.8, 90, -40, [0.5, 0, 0])
        else:
            self._p = bc.BulletClient(connection_mode=p.DIRECT)
        self._physics_client_id = self._p._client
        self._p.resetSimulation()
        # self._p.setTimeStep(self.dt)
        self._p.setRealTimeSimulation(0)
        self._p.setGravity(0, 0, -9.8)
        self._p.setPhysicsEngineParameter(enableConeFriction=1)


    def close_pb(self):
        """
        Kills created objects and closes pybullet simulator.
        """
        self._p.disconnect()
        self._physics_client_id = -1
        self._panda_id = -1


    def reset_robot(self, mu=0.5, sigma=0.01, task=None):
        """
        Reset robot for the environment. Called in reset() if loading robot for
        the 1st time, or in reset_task() if loading robot for the 2nd time.
        """
        if self._panda_id < 0:
            # self._p.URDF_USE_SELF_COLLISION 
            # | self._p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT 
            # | self._p.URDF_USE_MATERIAL_COLORS_FROM_MTL
            self._panda_id = self._p.loadURDF(
                self._panda_urdf_path,
                basePosition=[0, 0, 0],
                baseOrientation=[0, 0, 0, 1],
                useFixedBase=1,
                flags=self._p.URDF_USE_INERTIA_FROM_FILE
                )

            # Set friction coefficient for fingers
            self._p.changeDynamics(
                self._panda_id,
                self._left_finger_link_id,
                lateralFriction=mu,
                spinningFriction=sigma,
                frictionAnchor=1,
            )
            self._p.changeDynamics(
                self._panda_id,
                self._right_finger_link_id,
                lateralFriction=mu,
                spinningFriction=sigma,
                frictionAnchor=1,
            )

            # Create a constraint to keep the fingers centered (upper links)
            fingerGear = self._p.createConstraint(
                self._panda_id,
                self._left_finger_joint_id,
                self._panda_id,
                self._right_finger_joint_id,
                jointType=self._p.JOINT_GEAR,
                jointAxis=[1, 0, 0],
                parentFramePosition=[0, 0, 0],
                childFramePosition=[0, 0, 0])
            self._p.changeConstraint(fingerGear,
                                     gearRatio=-1,
                                     erp=0.1,
                                     maxForce=2 * self._max_finger_force)

            # Disable damping for all links
            for i in range(self._num_joint):
                self._p.changeDynamics(self._panda_id,
                                       i,
                                       linearDamping=0,
                                       angularDamping=0)

            # Measure EE joint
            self._p.enableJointForceTorqueSensor(self._panda_id, self._ee_joint_id, 1)

        # Solve ik if task specifies initial pose
        if hasattr(task, 'init_pose'):
            pos = task['init_pose'][:3]
            orn = task['init_pose'][3:]
            init_joint_angles = self.get_ik(pos, orn)
        else:
            init_joint_angles = self._default_joint_angles

        # Reset all joint angles
        self.reset_robot_joints(init_joint_angles)


    def reset_robot_joints(self, angles):
        """[summary]

        Args:
            angles ([type]): [description]
        """
        if len(angles) < self._num_joint:  # 7
            angles += [
                0, 0, self._finger_open_pos, 0.0,
                self._finger_open_pos, 0.0
            ]
        for i in range(self._num_joint):  # 13
            # print(self._p.getJointState(self._panda_id, i))
            # print(self._p.getJointInfo(self._panda_id, i))
            self._p.resetJointState(self._panda_id, i, angles[i])


    def reset_arm_joints_ik(self, pos, orn, gripper_closed=False):
        """[summary]

        Args:
            pos ([type]): [description]
            orn ([type]): [description]
            gripper_closed (bool, optional): [description]. Defaults to False.
        """
        joint_poses = self.get_ik(pos, orn)
        if gripper_closed:
            finger_pos = self._finger_close_pos
        else:
            finger_pos = self._finger_open_pos
        joint_poses = joint_poses[:7] + [
            0, 0, finger_pos, 0.00, finger_pos, 0.00
        ]
        self.reset_robot_joints(joint_poses)


    def grasp(self, target_vel=0):
        """
        Change gripper velocity direction
        """
        self._finger_cur_vel = target_vel


    def move_pose(
        self,
        absolute_pos=None,
        relative_pos=None,
        absolute_global_euler=None,  # preferred
        relative_global_euler=None,  # preferred
        relative_local_euler=None,  # not using
        absolute_global_quat=None,  # preferred
        relative_azi=None,  # for arm
        # relative_quat=None,  # never use relative quat
        num_steps=50,
        max_joint_vel=0.20,
        pos_gain=20,
        vel_gain=5,
        collision_force_threshold=0,
    ):

        # Get trajectory
        ee_pos, ee_quat = self._get_ee()

        # Determine target pos
        if absolute_pos is not None:
            target_pos = absolute_pos
        elif relative_pos is not None:
            target_pos = ee_pos + relative_pos
        else:
            target_pos = ee_pos

        # Determine target orn
        if absolute_global_euler is not None:
            target_orn = euler2quat(absolute_global_euler)
        elif relative_global_euler is not None:
            target_orn = quatMult(euler2quat(relative_global_euler), ee_quat)
        elif relative_local_euler is not None:
            target_orn = quatMult(ee_quat, euler2quat(relative_local_euler))
        elif absolute_global_quat is not None:
            target_orn = absolute_global_quat
        elif relative_azi is not None:
            # Extrinsic yaw
            target_orn = quatMult(euler2quat([relative_azi[0], 0, 0]), ee_quat)
            # Intrinsic pitch
            target_orn = quatMult(target_orn,
                                  euler2quat([0, relative_azi[1], 0]))
        # elif relative_quat is not None:
        # 	target_orn = quatMult(ee_quat, relative_quat)
        else:
            target_orn = np.array([1.0, 0., 0., 0.])

        # Get trajectory
        traj_pos = traj_time_scaling(start_pos=ee_pos,
                                     end_pos=target_pos,
                                     num_steps=num_steps)

        # Run steps
        collision = False
        num_steps = len(traj_pos)
        for step in range(num_steps):

            # Get joint velocities from error tracking control, takes 0.2ms
            joint_dot = self.traj_tracking_vel(target_pos=traj_pos[step],
                                               target_quat=target_orn,
                                               pos_gain=pos_gain,
                                               vel_gain=vel_gain)

            # Send velocity commands to joints
            for i in range(self._num_joint_arm):
                self._p.setJointMotorControl2(self._panda_id,
                                              i,
                                              self._p.VELOCITY_CONTROL,
                                              targetVelocity=joint_dot[i],
                                              force=self._max_joint_force[i],
                                              maxVelocity=max_joint_vel)

            # Keep gripper current velocity
            self._p.setJointMotorControl2(self._panda_id,
                                          self._left_finger_joint_id,
                                          self._p.VELOCITY_CONTROL,
                                          targetVelocity=self._finger_cur_vel,
                                          force=self._max_finger_force,
                                          maxVelocity=0.10)
            self._p.setJointMotorControl2(self._panda_id,
                                          self._right_finger_joint_id,
                                          self._p.VELOCITY_CONTROL,
                                          targetVelocity=self._finger_cur_vel,
                                          force=self._max_finger_force,
                                          maxVelocity=0.10)

            # Check contact
            if collision_force_threshold > 0:
                fm = np.array(self._p.getJointState(self._panda_id, self._ee_joint_id)[2])
                if np.any(fm[:3] > collision_force_threshold):
                    collision = True

            # Step simulation, takes 1.5ms
            self._p.stepSimulation()
        return collision


    def move_vel(self, target_lin_vel, target_ang_vel, num_steps, 
                        check_obj_between_finger=False,
                        grasp_vel=None,
                        init_quat=None,
                        max_roll=np.pi/4,
                        max_pitch=np.pi/4,
                        roll_spring_threshold=np.pi/36,
                        pitch_spring_threshold=np.pi/36,
                        max_roll_vel=np.pi/4,
                        max_pitch_vel=np.pi/4,
                        apply_grasp_threshold=None,
                        z_vel_spring_threshold=0.05,
                        z_vel_max=0.1):
        target_vel = np.hstack((target_lin_vel, target_ang_vel))

        ray_queue = deque([0 for _ in range(10)], maxlen=10)
        for _ in range(num_steps):
            joint_poses = list(
                np.hstack((self._get_arm_joints(), np.array([0, 0]))))  # add fingers
            ee_state = self._p.getLinkState(self._panda_id,
                                            self._ee_link_id,
                                            computeLinkVelocity=1,
                                            computeForwardKinematics=1)

            # Apply spring to z velocity if close to table
            finger_z = self._get_lowerest_pos()[2]
            if finger_z < z_vel_spring_threshold and target_vel[2] < 0:
                min_vel = -z_vel_max*(finger_z / z_vel_spring_threshold)
                target_vel[2] = max(min_vel, target_vel[2])

            # Apply spring to roll / pitch if close to limit
            if init_quat is not None:
                _, ee_quat = self._get_ee()
                ee_quat_diff = quatMult(ee_quat, quatInverse(init_quat))
                _, pitch_diff, roll_diff = quat2euler(ee_quat_diff)
                if roll_diff > (max_roll-roll_spring_threshold):
                    max_vel = max_roll_vel*((max_roll-roll_diff) / roll_spring_threshold)
                    target_vel[3] = min(max_vel, target_vel[3])
                if roll_diff < (-max_roll+roll_spring_threshold):
                    min_vel = -max_roll_vel*((max_roll+roll_diff) / roll_spring_threshold)
                    target_vel[3] = max(min_vel, target_vel[3])
                if pitch_diff > (max_pitch-pitch_spring_threshold):
                    max_vel = max_pitch_vel*((max_pitch-pitch_diff) / pitch_spring_threshold)
                    target_vel[4] = min(max_vel, target_vel[4])
                if pitch_diff < (-max_pitch+pitch_spring_threshold):
                    min_vel = -max_pitch_vel*((max_pitch+pitch_diff) / pitch_spring_threshold)
                    target_vel[4] = max(min_vel, target_vel[4])

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

            # Apply grasp if specified
            if apply_grasp_threshold is not None:
                finger_z = self._get_lowerest_pos()[2]
                if finger_z > apply_grasp_threshold and not self.grasp_executed:
                    self.grasp(self._finger_open_vel)
                else:
                    self.grasp(self._finger_close_vel)
                    self.grasp_executed = True
            elif grasp_vel is not None:
                self.grasp(grasp_vel)

            # Object between finger
            # elif check_obj_between_finger and self._check_obj_between_finger():
            #     self.grasp(self._finger_close_vel)
            #     self.grasp_executed = True
            #     print('here')
            # else:
            #     print('open')
            #     self.grasp(self._finger_open_vel)

            # Step simulation, takes 1.5ms
            # import time
            # s1 = time.time()
            self._p.stepSimulation()
            # print(time.time()-s1)
            # for info in self._p.getContactPoints(self.block_id, self.peg_id):
            #     print(info[8], info[9])
            # exit()
            # print(
            #     p.getLinkState(self._pandaId,
            #                    self._panda.pandaEndEffectorLinkIndex,
            #                    computeLinkVelocity=1)[6])
            # print(p.getBaseVelocity(objId)[1])
            # print("===")
        return 1


    def traj_tracking_vel(self,
                          target_pos,
                          target_quat,
                          pos_gain=20,
                          vel_gain=5):  #Change gains based off mouse
        ee_pos, ee_quat = self._get_ee()

        ee_pos_error = target_pos - ee_pos
        # ee_orn_error = log_rot(quat2rot(target_quat)@(quat2rot(ee_quat).T))  # in spatial frame
        ee_orn_error = log_rot(
            quat2rot(target_quat).dot(
                (quat2rot(ee_quat).T)))  # in spatial frame

        joint_poses = list(
            np.hstack((self._get_arm_joints(), np.array([0, 0]))))  # add fingers
        ee_state = self._p.getLinkState(self._panda_id,
                                        self._ee_link_id,
                                        computeLinkVelocity=1,
                                        computeForwardKinematics=1)
        # Get the Jacobians for the CoM of the end-effector link. Note that in this example com_rot = identity, and we would need to use com_rot.T * com_trn. The localPosition is always defined in terms of the link frame coordinates.
        zero_vec = list(np.zeros_like(joint_poses))
        jac_t, jac_r = self._p.calculateJacobian(
            self._panda_id, self._ee_link_id, ee_state[2], joint_poses,
            zero_vec, zero_vec)  # use localInertialFrameOrientation
        jac_sp = full_jacob_pb(
            jac_t, jac_r)[:, :7]  # 6x10 -> 6x7, ignore last three columns
        try:
            joint_dot = np.linalg.pinv(jac_sp).dot((np.hstack(
                (pos_gain * ee_pos_error,
                 vel_gain * ee_orn_error)).reshape(6, 1)))  # pseudo-inverse
        except np.linalg.LinAlgError:
            joint_dot = np.zeros((7, 1))
        return joint_dot


    ################# Observation #################

    def _get_obs(self):
        raise NotImplementedError


    def get_overhead_obs(self, camera_param):
        far = 1000.0
        near = 0.01
        camera_pos = np.array(camera_param['pos'])
        rot_matrix = quat2rot(self._p.getQuaternionFromEuler(camera_param['euler']))
        init_camera_vector = (0, 0, 1)  # z-axis
        init_up_vector = (1, 0, 0)  # x-axis
        camera_vector = rot_matrix.dot(init_camera_vector)
        up_vector = rot_matrix.dot(init_up_vector)

        view_matrix = self._p.computeViewMatrix(
            camera_pos, camera_pos + 0.1 * camera_vector, up_vector)
        projection_matrix = self._p.computeProjectionMatrixFOV(
            fov=camera_param['fov'],
            aspect=camera_param['aspect'],
            nearVal=near,
            farVal=far)

        _, _, rgb, depth, _ = self._p.getCameraImage(
            camera_param['img_w'],
            camera_param['img_h'],
            view_matrix,
            projection_matrix,
            flags=self._p.ER_NO_SEGMENTATION_MASK)
        out = []
        if camera_param['use_depth']:
            depth = far * near / (far - (far - near) * depth)
            depth = (camera_param['overhead_max_depth'] - depth) / (camera_param['overhead_max_depth'] - camera_param['overhead_min_depth'])
            depth = depth.clip(min=0., max=1.)
            if camera_param['save_byte']:
                depth = np.uint8(depth * 255)
            out += [depth[np.newaxis]]
        if camera_param['use_rgb']:
            # TODO: save_byte option
            rgb = rgba2rgb(rgb).transpose(2, 0, 1)
            out += [rgb]
        out = np.concatenate(out)
        return out  # uint8


    def get_wrist_obs(self, camera_param):    # todo: use dict for params
        ee_pos, ee_quat = self._get_ee()
        rot_matrix = quat2rot(ee_quat)
        camera_pos = ee_pos + rot_matrix.dot(camera_param['wrist_offset'])
        # plot_frame_pb(camera_pos, ee_orn)

        # Initial vectors
        init_camera_vector = (0, 0, 1)  # z-axis
        init_up_vector = (1, 0, 0)  # x-axis

        # Rotated vectors
        camera_vector = rot_matrix.dot(init_camera_vector)
        up_vector = rot_matrix.dot(init_up_vector)
        view_matrix = self._p.computeViewMatrix(
            camera_pos, camera_pos + 0.1 * camera_vector, up_vector)

        # Get Image
        far = 1000.0
        near = 0.01
        projection_matrix = self._p.computeProjectionMatrixFOV(
            fov=camera_param['fov'],
            aspect=camera_param['aspect'],
            nearVal=near,
            farVal=far)
        _, _, rgb, depth, _ = self._p.getCameraImage(
            camera_param['img_w'],
            camera_param['img_h'],
            view_matrix,
            projection_matrix,
            flags=self._p.ER_NO_SEGMENTATION_MASK)
        out = []
        if self.use_depth:
            depth = far * near / (far - (far - near) * depth)
            depth = (camera_param['wrist_max_depth'] - depth) / camera_param['wrist_max_depth']
            # depth += np.random.normal(loc=0, scale=0.02, size=depth.shape)    #todo: use self.rng
            depth = depth.clip(min=0., max=1.)
            depth = np.uint8(depth * 255)
            out += [depth[np.newaxis]]
        if self.use_rgb:
            rgb = rgba2rgb(rgb).transpose(2, 0, 1)  # store as uint8
            # rgb_mask = np.uint8(np.random.choice(np.arange(0,2), size=rgb.shape[1:], replace=True, p=[0.95, 0.05]))
            # rgb_random = np.random.randint(0, 256, size=rgb.shape[1:], dtype=np.uint8)  #todo: use self.rng
            # rgb_mask *= rgb_random
            # rgb = np.where(rgb_mask > 0, rgb_mask, rgb)
            out += [rgb]
        out = np.concatenate(out)
        return out


    # def get_pixel_2_xy(self, camera_param):
    #     far = 1000.0
    #     near = 0.01
    #     camera_pos = np.array(camera_param['pos'])
    #     rot_matrix = quat2rot(self._p.getQuaternionFromEuler(camera_param['euler']))
    #     init_camera_vector = (0, 0, 1)  # z-axis
    #     init_up_vector = (1, 0, 0)  # x-axis
    #     camera_vector = rot_matrix.dot(init_camera_vector)
    #     up_vector = rot_matrix.dot(init_up_vector)

    #     view_matrix = self._p.computeViewMatrix(
    #         camera_pos, camera_pos + 0.1 * camera_vector, up_vector)
    #     projection_matrix = self._p.computeProjectionMatrixFOV(
    #         fov=camera_param['fov'],
    #         aspect=camera_param['aspect'],
    #         nearVal=near,
    #         farVal=far)

    #     _, _, _, depth_buffer, _ = self._p.getCameraImage(
    #         camera_param['img_w'],
    #         camera_param['img_h'],
    #         view_matrix,
    #         projection_matrix,
    #         flags=self._p.ER_NO_SEGMENTATION_MASK)

    #     # Assume top-down, positive x in the world is left in the image
    #     # camera_target = np.copy(camera_pos)    # assume top down
    #     # camera_target[-1] = 0
    #     camera_param['far'] = far
    #     camera_param['near'] = near
    #     camera_param['cam_forward'] = (0.0, -1.0, -1.7881393432617188e-06)
    #     camera_param['horizon'] = (20000.0, -0.0, 0.0)
    #     camera_param['vertical'] = (0.0, 0.035762786865234375, -20000.0)
    #     camera_param['dist'] = 0.4000000059604645 
    #     camera_param['camera_target'] = (0.5, 0.0, 0.0)
    #     return depth_pixel_2_xy(depth_buffer, param=camera_param)


    ################# Misc info #################

    def get_ik(self, pos, orn):
        # Null-space IK not working now - Need to manually set joints to rest pose
        self.reset_robot_joints(self._joint_rest_pose)
        joint_poses = self._p.calculateInverseKinematics(
                self._panda_id,
                self._ee_link_id,
                pos,
                orn,
                jointDamping=self._jd,  # damping required - not sure why
                # lowerLimits=self._joint_lower_limit,
                # upperLimits=self._joint_upper_limit,
                # jointRanges=self._joint_range,
                # restPoses=self._joint_rest_pose,
                residualThreshold=1e-4,
                # solver=self._p.IK_SDLS,
                # maxNumIterations=1e5
                )
        return list(joint_poses)


    def _get_lowerest_pos(self):
        """Assume fingertips"""
        pos, quat = self._get_ee()
        joint = self._get_gripper_joint()[0]
        pos_1 = pos + quat2rot(quat)@np.array([0.0, joint, 0.155])
        pos_2 = pos + quat2rot(quat)@np.array([0.0, -joint, 0.155])
        if pos_1[2] < pos_2[2]:
            return pos_1
        else:
            return pos_2


    def _get_ee(self):
        info = self._p.getLinkState(self._panda_id, self._ee_link_id)
        return np.array(info[4]), np.array(info[5])


    def _get_arm_joints(self):  # use list
        info = self._p.getJointStates(self._panda_id,
                                      range(self._num_joint_arm))
        return np.array([x[0] for x in info])


    def _get_gripper_joint(self):
        info = self._p.getJointState(
            self._panda_id, self._left_finger_joint_id)  # assume symmetrical
        return np.array(info[0]), np.array(info[1])


    def _get_left_finger(self):
        info = self._p.getLinkState(self._panda_id, self._left_finger_link_id)
        return np.array(info[4]), np.array(info[5])


    def _get_right_finger(self):
        info = self._p.getLinkState(self._panda_id, self._right_finger_link_id)
        return np.array(info[4]), np.array(info[5])


    def _get_state(self):
        ee_pos, ee_orn = self._get_ee()
        arm_joints = self._get_arm_joints()  # angles only
        return np.hstack((ee_pos, ee_orn, arm_joints))


    def _check_obj_contact(self, obj_id, both=False):
        left_contacts, right_contacts = self._get_finger_contact(obj_id)
        if both:
            if len(left_contacts) > 0 and len(right_contacts) > 0:
                return 1
        else:
            if len(left_contacts) > 0 or len(right_contacts) > 0:
                return 1
        return 0


    def _get_finger_contact(self, obj_id, min_force=1e-1):
        num_joint = self._p.getNumJoints(obj_id)
        link_all = [-1] + [*range(num_joint)]
        left_contacts = []
        right_contacts = []
        for link_id in link_all:
            left_contact = self._p.getContactPoints(
                self._panda_id,
                obj_id,
                linkIndexA=self._left_finger_link_id,
                linkIndexB=link_id)
            right_contact = self._p.getContactPoints(
                self._panda_id,
                obj_id,
                linkIndexA=self._right_finger_link_id,
                linkIndexB=link_id)
            left_contact = [i for i in left_contact if i[9] > min_force]
            right_contact = [i for i in right_contact if i[9] > min_force]
            left_contacts += left_contact
            right_contacts += right_contact
        return left_contacts, right_contacts


    def _get_finger_force(self, obj_id):
        left_contacts, right_contacts = self._get_finger_contact(obj_id)
        left_force = np.zeros((3))
        right_force = np.zeros((3))
        for i in left_contacts:
            left_force += i[9] * np.array(i[7]) + i[10] * np.array(
                i[11]) + i[12] * np.array(i[13])
        for i in right_contacts:
            right_force += i[9] * np.array(i[7]) + i[10] * np.array(
                i[11]) + i[12] * np.array(i[13])
        left_normal_mag = sum([i[9] for i in left_contacts])
        right_normal_mag = sum([i[9] for i in right_contacts])
        num_left_contact = len(left_contacts)
        num_right_contact = len(right_contacts)

        if num_left_contact < 1 or num_right_contact < 1:
            return None
        else:
            return left_force, right_force, \
             np.array(left_contacts[0][6]), np.array(right_contacts[0][6]), \
             left_normal_mag, right_normal_mag


    def _check_hold_object(self, obj_id, min_force=10.0):
        left_contacts, right_contacts = self._get_finger_contact(obj_id)
        left_normal_mag = sum([i[9] for i in left_contacts])
        right_normal_mag = sum([i[9] for i in right_contacts])
        return left_normal_mag > min_force and right_normal_mag > min_force


    def _get_min_dist_from_finger(self, obj_id, max_dist=0.2):
        info_left = self._p.getClosestPoints(self._panda_id, obj_id, 
                                        distance=max_dist, 
                                        linkIndexA=self._left_finger_link_id)
        info_right = self._p.getClosestPoints(self._panda_id, obj_id, 
                                        distance=max_dist, 
                                        linkIndexA=self._right_finger_link_id)
        infos = info_left + info_right
        dists = []
        for info in infos:
            finger_pos = info[5]
            obj_pos = info[6]
            dists += [np.linalg.norm(np.array(finger_pos)-np.array(obj_pos))]
        if len(dists) == 0:
            return max_dist
        else:
            return min(dists)


    def _get_min_dist_between_obj(self, a_id, b_id, max_dist=0.2):
        """Consider all links in both objects"""
        infos = self._p.getClosestPoints(a_id, b_id, 
                                        distance=max_dist)
        dists = []
        for info in infos:
            a_pos = info[5]
            b_pos = info[6]
            dists += [np.linalg.norm(np.array(a_pos)-np.array(b_pos))]
        if len(dists) == 0:
            return max_dist
        else:
            return min(dists)


    def _check_obj_between_finger(self):
        pos, quat = self._get_ee()
        joint = self._get_gripper_joint()[0]
        rayFrom = pos + quat2rot(quat)@np.array([0.015, joint, 0.15])
        rayTo = pos + quat2rot(quat)@np.array([0.015, -joint, 0.15])
        rayOutput1 = self._p.rayTest(
            rayFrom,
            rayTo,
        )
        rayOutput1 = [out for out in rayOutput1 if out[0] == self.obj_id]
        # self._p.addUserDebugLine(
        #             rayFrom, rayTo, lineColorRGB=[0,0.5,0.0], lineWidth=2
        #         )
        rayFrom = pos + quat2rot(quat)@np.array([-0.015, joint, 0.15])
        rayTo = pos + quat2rot(quat)@np.array([-0.015, -joint, 0.15])
        rayOutput2 = self._p.rayTest(
            rayFrom,
            rayTo,
        )  # 1st hit for each ray
        rayOutput2 = [out for out in rayOutput2 if out[0] == self.obj_id]
        # print(len(rayOutput1), len(rayOutput2))
        # self._p.addUserDebugLine(
        #             rayFrom, rayTo, lineColorRGB=[0.5,0.0,0.0], lineWidth=2
        #         )
        return len(rayOutput1) > 0 and len(rayOutput2) > 0
