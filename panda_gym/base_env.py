from abc import ABC, abstractmethod
import pathlib
import numpy as np
import torch
import gym
import pybullet as p
from pybullet_utils import bullet_client as bc

from alano.utils.save_init_args import save__init__args
from alano.geometry.camera import rgba2rgb
from alano.geometry.transform import quat2rot, euler2quat


def normalize_action(action, lower, upper):
    # assume action in [-1,1]
    return (action+1)/2*(upper-lower) + lower

class BaseEnv(gym.Env, ABC):
    def __init__(self,
                 task=None,
                 renders=False,
                 use_rgb=False,
                 use_depth=True,
                 camera_params=None,
                #  finger_type='drake'
                 ):
        """
        Args:
            task (str, optional): the name of the task. Defaults to None.
            render (bool, optional): whether to render the environment.
                Defaults to False.
        """
        super(BaseEnv, self).__init__()
        if task is None:
            task = {}
            task['init_joint_angles'] = [0, 0.277, 0, -2.813, 0, 3.483, 0.785]
            task['obj_half_dim'] = [0.03,0.03,0.03]
            task['obj_mass'] = 0.1
            task['obj_pos'] = [0.5, 0.0, 0.03]
            task['obj_yaw'] = 0
            task['obj_com_offset'] = [0,0,0]
        if camera_params is None:
            camera_params = {}
            camera_height = 0.40
            camera_params['pos'] = np.array([1.0, 0, camera_height])
            camera_params['euler'] = [0, -3*np.pi/4, 0] # extrinsic - x up, z forward
            camera_params['img_w'] = 64
            camera_params['img_h'] = 64
            camera_params['aspect'] = 1
            camera_params['fov'] = 70    # vertical fov in degrees
            camera_params['max_depth'] = camera_height
        save__init__args(locals())  # save all class variables

        # PyBullet instance
        self._p = None
        self._physics_client_id = -1  # PyBullet
        self._panda_id = -1

        # Set up observation and action space for Gym
        _num_img_channel = 0
        if use_rgb:
            _num_img_channel += 3  # RGB
        if use_depth:
            _num_img_channel += 1  # D only
        self._camera_params = camera_params

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
        self._max_finger_force = 20.0
        # self.maxFingerForce = 35  # office documentation says 70N continuous force
        self._jd = [
            0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001,
            0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001
        ]  # joint damping coefficient
        self._joint_upper_limit = [2.90, 1.76, 2.90, -0.07, 2.90, 3.75, 2.90]
        self._joint_lower_limit = [
            -2.90, -1.76, -2.90, -3.07, -2.90, -0.02, -2.90
        ]
        self._joint_range = [5.8, 3.5, 5.8, 3, 5.8, 3.8, 5.8]
        self._joint_rest_pose = [0, -1.4, 0, -1.4, 0, 1.2, 0]
        self._joint_max_vel = np.array([2.0, 2.0, 2.0, 2.0, 2.5, 2.5,
                                        2.5])  # actually 2.175 and 2.61
        self._finger_open_pos = 0.04
        self._finger_close_pos = 0.0
        # Initialize current finger pos/vel
        self._finger_cur_pos = 0.04
        self._finger_cur_vel = 0.05

    @property
    @abstractmethod
    def action_dim(self):
        """
        Dimension of robot action
        """
        raise NotImplementedError

    # @property
    # def init_joint_angles(self):
    #     """
    #     Initial joint angles for the task
    #     """
    #     raise NotImplementedError

    # @property
    # def up_joint_angles(self):
    #     """[0.5, 0, 0.5], straight down - avoid mug hitting gripper when dropping
    #     """
    #     return [0, 1.643, 0, 1.167, 0, 0.476, 0.785, 0, 0,
    #         self._finger_open_pos, 0.00, self._finger_open_pos, 0.00
        # ]

    @abstractmethod
    def step(self):
        """
        Gym style step function. Apply action, move robot, get observation,
        calculate reward, check if done.
        """
        raise NotImplementedError

    @abstractmethod
    def report(self):
        """
        Print information of robot dynamics and observation.
        """
        raise NotImplementedError

    @abstractmethod
    def visualize(self):
        """
        Visualize trajectories and value functions.
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self, task=None):
        """
        Reset the environment, including robot state, task, and obstacles.
        Initialize pybullet client if 1st time.
        """
        raise NotImplementedError

    @abstractmethod
    def reset_task(self):
        """
        Reset the task for the environment.
        """
        raise NotImplementedError

    @property
    def state(self):
        return

    def seed(self, seed=0):
        """
        Set the seed of the environment. Should be called after action_sapce is
        defined.

        Args:
            seed (int, optional): random seed value. Defaults to 0.
        """
        self.seed_val = seed
        self.rng = np.random.default_rng(seed=self.seed_val)
        torch.manual_seed(self.seed_val)
        torch.cuda.manual_seed(self.seed_val)
        torch.cuda.manual_seed_all(
            self.seed_val)  # if you are using multi-GPU.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        # self.np_random, seed = gym.utils.seeding.np_random(seed)
        # return [seed]

    def init_pb(self):
        """
        Initialize PyBullet environment.
        """
        if self.renders:
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
        Kills created obkects and closes pybullet simulator.
        """
        self._p.disconnect()
        self._physics_client_id = -1
        self._panda_id = -1

    def reset_robot(self, mu=0.5, sigma=0.01, init_joint_angles=[
            0, 0.277, 0, -2.813, 0, 3.483, 0.785]):
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

        # Reset all joint angles
        self.reset_robot_joints(init_joint_angles)

        # Keep gripper open
        self.grasp(target_vel=0.10)

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

    @abstractmethod
    def move(self):
        """
        Move robot to the next state; returns next state
        """
        raise NotImplementedError

    def grasp(self, target_vel=0):
        """
        Change gripper velocity direction
        """
        self._finger_cur_vel = target_vel
        # if target_vel > 1e-2 or target_vel < -1e-2:
        #     self._finger_cur_vel = target_vel
        # else:
        #     if self._finger_cur_vel > 0.0:
        #         self._finger_cur_vel = -0.05
        #     else:
        #         self._finger_cur_vel = 0.05

    ################# Obs #################

    @abstractmethod
    def _get_obs(self):
        raise NotImplementedError

    def get_overhead_obs(self, camera_params):
        far = 1000.0
        near = 0.01
        camera_pos = camera_params['pos']
        rot_matrix = quat2rot(self._p.getQuaternionFromEuler(camera_params['euler']))
        init_camera_vector = (0, 0, 1)  # z-axis
        init_up_vector = (1, 0, 0)  # x-axis
        camera_vector = rot_matrix.dot(init_camera_vector)
        up_vector = rot_matrix.dot(init_up_vector)

        view_matrix = self._p.computeViewMatrix(
            camera_pos, camera_pos + 0.1 * camera_vector, up_vector)
        projection_matrix = self._p.computeProjectionMatrixFOV(
            fov=camera_params['fov'],
            aspect=camera_params['aspect'],
            nearVal=near,
            farVal=far)

        _, _, rgb, depth, _ = self._p.getCameraImage(
            camera_params['img_w'],
            camera_params['img_h'],
            view_matrix,
            projection_matrix,
            flags=self._p.ER_NO_SEGMENTATION_MASK)
        out = []
        if self.use_depth:
            depth = far * near / (far - (far - near) * depth)
            depth = (camera_params['max_depth'] - depth) / camera_params['max_depth']
            depth = depth.clip(min=0., max=1.)
            depth = np.uint8(depth * 255)
            out += [depth[np.newaxis]]
        if self.use_rgb:
            rgb = rgba2rgb(rgb).transpose(2, 0, 1)
            out += [rgb]
        out = np.concatenate(out)
        return out  # uint8

    def get_wrist_obs(self):    # todo: use dict for params
        """Wrist camera image
        """
        ee_pos, ee_quat = self._get_ee()
        rot_matrix = quat2rot(ee_quat)
        camera_pos = ee_pos + rot_matrix.dot(self.camera_wrist_offset)
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
            fov=self.camera_fov,
            aspect=self.camera_aspect,
            nearVal=near,
            farVal=far)
        _, _, rgb, depth, _ = self._p.getCameraImage(
            self.img_w,
            self.img_h,
            view_matrix,
            projection_matrix,
            flags=self._p.ER_NO_SEGMENTATION_MASK)
        out = []
        if self.use_depth:
            depth = far * near / (far - (far - near) * depth)
            depth = (self.camera_max_depth - depth) / self.camera_max_depth
            depth += np.random.normal(loc=0, scale=0.02, size=depth.shape)    #todo: use self.rng
            depth = depth.clip(min=0., max=1.)
            depth = np.uint8(depth * 255)
            out += [depth[np.newaxis]]
        if self.use_rgb:
            rgb = rgba2rgb(rgb).transpose(2, 0, 1)  # store as uint8
            rgb_mask = np.uint8(np.random.choice(np.arange(0,2), size=rgb.shape[1:], replace=True, p=[0.95, 0.05]))
            rgb_random = np.random.randint(0, 256, size=rgb.shape[1:], dtype=np.uint8)  #todo: use self.rng
            rgb_mask *= rgb_random
            rgb = np.where(rgb_mask > 0, rgb_mask, rgb)
            out += [rgb]
        out = np.concatenate(out)

        return out

    ################# Get info #################

    def get_ik(self, pos, orn):
        joint_poses = self._p.calculateInverseKinematics(
                self._panda_id,
                self._ee_link_id,
                pos,
                orn,
                jointDamping=self._jd,
                lowerLimits=self._joint_lower_limit,
                upperLimits=self._joint_upper_limit,
                jointRanges=self._joint_range,
                restPoses=self._joint_rest_pose,
                residualThreshold=1e-4,
                # maxNumIterations=1e5
                )
        return list(joint_poses)

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
