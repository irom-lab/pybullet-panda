from abc import ABC, abstractmethod
import numpy as np
import torch
import gym
import pybullet as p
from pybullet_utils import bullet_client as bc

from alano.utils.save_init_args import save__init__args


class BaseEnv(gym.Env, ABC):
    def __init__(self,
                 task=None,
                 renders=False,
                 img_h=128,
                 img_w=128,
                 use_rgb=False,
                 use_depth=True,
                 max_steps_train=100,
                 max_steps_eval=100,
                 done_type='fail',
                 finger_type='long'):
        """
        Args:
            task (str, optional): the name of the task. Defaults to None.
            img_h (int, optional): the height of the image. Defaults to 128.
            img_w (int, optional): the width of the image. Defaults to 128.
            use_rgb (bool, optional): whether to use RGB image. Defaults to
                True.
            render (bool, optional): whether to render the environment.
                Defaults to False.
            max_steps_train (int, optional): the maximum number of steps to
                train. Defaults to 100.
            max_steps_eval (int, optional): the maximum number of steps to
                evaluate. Defaults to 100.
            done_type (str, optional): the type of the done. Defaults to
                'fail'.
        """
        super(BaseEnv, self).__init__()
        save__init__args(locals())  # save all class variables

        # PyBullet instance
        self._p = None
        self._physics_client_id = -1  # PyBullet
        self._panda_id = -1

        # Flag for train/eval
        self.set_train_mode()

        # Set up observation and action space for Gym
        _num_img_channel = 0
        if use_rgb:
            _num_img_channel += 3  # RGB
        if use_depth:
            _num_img_channel += 1  # D only
        self.observation_space = gym.spaces.Box(low=np.float32(0.),
                                                high=np.float32(1.),
                                                shape=(_num_img_channel, img_h,
                                                       img_w))

        # Panda config
        if finger_type is None:
            _finger_name = 'panda_arm_finger_orig'
        elif finger_type == 'long':
            _finger_name = 'panda_arm_finger_long'
        elif finger_type == 'wide_curved':
            _finger_name = 'panda_arm_finger_wide_curved'
        elif finger_type == 'wide_flat':
            _finger_name = 'panda_arm_finger_wide_flat'
        else:
            raise NotImplementedError
        self._panda_urdf_path = f'data/franka/{_finger_name}.urdf'
        self._num_joint = 13
        self._num_joint_arm = 7  # Number of joints in arm (not counting hand)
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
        self._finger_cur_pos = 0.04  # current
        self._finger_cur_vel = 0.05

    @property
    @abstractmethod
    def action_dim(self):
        """
        Dimension of robot action
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def init_joint_angles(self):
        """
        Initial joint angles for the task
        """
        raise NotImplementedError

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

    def set_train_mode(self):
        """
        Set the environment to train mode.
        """
        self.flag_train = True
        self.max_steps = self.max_steps_train

    def set_eval_mode(self):
        """
        Set the environment to eval mode.
        """
        self.flag_train = False
        self.max_steps = self.max_steps_eval

    def seed(self, seed=0):
        """
        Set the seed of the environment. Should be called after action_sapce is
        defined.

        Args:
            seed (int, optional): random seed value. Defaults to 0.
        """
        self.seed_val = seed
        self.action_space.seed(self.seed_val)
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

    def reset_robot(self, mu=0.5, sigma=0.01):
        """
        Reset robot for the environment. Called in reset() if loading robot for
        the 1st time, or in reset_task() if loading robot for the 2nd time.
        """
        if self._panda_id < 0:
            self._panda_id = self._p.loadURDF(
                self._panda_urdf_path,
                basePosition=[0, 0, 0],
                baseOrientation=[0, 0, 0, 1],
                useFixedBase=1,
                flags=(self._p.URDF_USE_SELF_COLLISION
                       and self._p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT))

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

        # Reset all joint angles
        self.reset_robot_joints(self.init_joint_angles)

        # Kep gripper open
        self.grasp(target_vel=0.10)

    def reset_robot_joints(self, angles):
        """[summary]

        Args:
            angles ([type]): [description]
        """
        if len(angles) < self._num_joint:  # 7
            angles += [
                0, -np.pi / 4, self._finger_open_pos, 0.0,
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
        joint_poses = list(
            self._p.calculateInverseKinematics(
                self._panda_id,
                self._ee_link_id,
                pos,
                orn,
                jointDamping=self._jd,
                lowerLimits=self._joint_lower_limit,
                upperLimits=self._joint_upper_limit,
                jointRanges=self._joint_range,
                restPoses=self._joint_rest_pose,
                residualThreshold=1e-4))
        #    , maxNumIterations=1e5))
        if gripper_closed:
            finger_pos = self._finger_close_pos
        else:
            finger_pos = self._finger_open_pos
        joint_poses = joint_poses[:7] + [
            0, -np.pi / 4, finger_pos, 0.00, finger_pos, 0.00
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
        if target_vel > 1e-2 or target_vel < -1e-2:
            self._finger_cur_vel = target_vel
        else:
            if self._finger_cur_vel > 0.0:
                self._finger_cur_vel = -0.05
            else:
                self._finger_cur_vel = 0.05

    ################# Get info #################

    @abstractmethod
    def _get_obs(self):
        raise NotImplementedError

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
        left_contacts = self._p.getContactPoints(
            self._panda_id,
            obj_id,
            linkIndexA=self._left_finger_link_id,
            linkIndexB=-1)
        right_contacts = self._p.getContactPoints(
            self._panda_id,
            obj_id,
            linkIndexA=self._right_finger_link_id,
            linkIndexB=-1)
        left_contacts = [i for i in left_contacts if i[9] > min_force]
        right_contacts = [i for i in right_contacts if i[9] > min_force]
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
        # print((left_normal_mag, right_normal_mag, num_left_contact, num_right_contact))

        if num_left_contact < 1 or num_right_contact < 1:
            # print((numLeftContact, numRightContact))
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

