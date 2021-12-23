from abc import ABC, abstractmethod
import numpy as np
import torch
import pybullet as p
from pybullet_utils import bullet_client as bc
import os
import gym

from panda.utils import save__init__args
from panda.util_geom import euler2quat, quat2rot, quatMult, log_rot


class BaseEnv(gym.Env, ABC):
    def __init__(self,
                 task=None,
                 render=False,
                 img_H=128,
                 img_W=128,
                 use_rgb=False,
                 max_steps_train=100,
                 max_steps_eval=100,
                 done_type='fail',
                 finger_type='long'):
        """
        Args:
            task (str, optional): the name of the task. Defaults to None.
            img_H (int, optional): the height of the image. Defaults to 128.
            img_W (int, optional): the width of the image. Defaults to 128.
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

        # Flag for train/eval
        self.set_train_mode()

        # Set up observation and action space for Gym
        if use_rgb:
            self.num_img_channel = 3  # RGB
        else:
            self.num_img_channel = 1  # D only
        self.observation_space = panda_gym.spaces.Box(
            low=np.float32(0.),
            high=np.float32(1.),
            shape=(self.num_img_channel, img_H, img_W))

        # Panda config
        if finger_type is None:
            self.finger_name = 'panda_arm_finger_orig'
        elif finger_type == 'long':
            self.finger_name = 'panda_arm_finger_long'
        elif finger_type == 'wide_curved':
            self.finger_name = 'panda_arm_finger_wide_curved'
        elif finger_type == 'wide_flat':
            self.finger_name = 'panda_arm_finger_wide_flat'
        else:
            raise NotImplementedError
        self.urdfRootPath = os.path.join(
            os.path.dirname(__file__),
            f'geometry/franka/{self.finger_name}.urdf')

        self.numJointsArm = 7  # Number of joints in arm (not counting hand)
        self.pandaEndEffectorLinkIndex = 8  # hand, index=7 is link8 (virtual one)
        self.pandaLeftFingerLinkIndex = 10  # lower
        self.pandaRightFingerLinkIndex = 12
        self.pandaLeftFingerJointIndex = 9
        self.pandaRightFingerJointIndex = 11
        self.maxJointForce = [87, 87, 87, 87, 12, 12, 12]  # from website
        self.maxFingerForce = 20.0
        # self.maxFingerForce = 35  # office documentation says 70N continuous force
        self.jd = [
            0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001,
            0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001
        ]  # joint damping coefficient
        self.jointUpperLimit = [2.90, 1.76, 2.90, -0.07, 2.90, 3.75, 2.90]
        self.jointLowerLimit = [
            -2.90, -1.76, -2.90, -3.07, -2.90, -0.02, -2.90
        ]
        self.jointRange = [5.8, 3.5, 5.8, 3, 5.8, 3.8, 5.8]
        self.jointRestPose = [0, -1.4, 0, -1.4, 0, 1.2, 0]
        self.jointMaxVel = np.array([2.0, 2.0, 2.0, 2.0, 2.5, 2.5,
                                     2.5])  # actually 2.175 and 2.61
        self.fingerOpenPos = 0.04
        self.fingerClosedPos = 0.0
        self.fingerCurPos = 0.04
        self.fingerCurVel = 0.05

        # TODO: attribute
        self.init_joint_angles = [
            0, -0.1, 0, -2, 0, 1.8, 0.785, 0, -np.pi / 4, self.fingerOpenPos,
            0.00, self.fingerOpenPos, 0.00
        ]

    @property
    @abstractmethod
    def action_dim(self):
        """
        Dimension of robot action
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

    # @abstractmethod
    def init_pb(self):
        """
        Initialize PyBullet environment.
        """
        if self.render:
            self._p = bc.BulletClient(connection_mode=p.GUI)
            self._p.resetDebugVisualizerCamera(0.8, 180, -45, [0.5, 0, 0])
        else:
            self._p = bc.BulletClient()
        self._physics_client_id = self._p._client
        self._p.resetSimulation()
        # self._p.setTimeStep(self.dt)
        self._p.setRealTimeSimulation(0)
        self._p.setGravity(0, 0, -9.8)
        self._p.setPhysicsEngineParameter(enableConeFriction=1)

    # @abstractmethod
    def close_pb(self):
        """
        Kills created obkects and closes pybullet simulator.
        """
        self._p.disconnect()
        self._physics_client_id = -1

    @abstractmethod
    def set_default_task(self):
        """
        Set default task for the environment.
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

    # @abstractmethod
    def reset_robot(self, mu=0.5, sigma=0.01):
        """
        Reset robot for the environment. Called in reset() if loading robot for
        the 1st time, or in reset_task() if loading robot for the 2nd time.
        """
        if self.panda_id < 0:
            self.panda_id = self._p.loadURDF(
                self.urdfRootPath,
                basePosition=[0, 0, 0],
                baseOrientation=[0, 0, 0, 1],
                useFixedBase=1,
                flags=(self._p.URDF_USE_SELF_COLLISION
                       and self._p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT))

            # Set friction coefficient for fingers
            self._p.changeDynamics(
                self.panda_id,
                self.pandaLeftFingerLinkIndex,
                lateralFriction=mu,
                spinningFriction=sigma,
                frictionAnchor=1,
            )
            self._p.changeDynamics(
                self.panda_id,
                self.pandaRightFingerLinkIndex,
                lateralFriction=mu,
                spinningFriction=sigma,
                frictionAnchor=1,
            )

            # Create a constraint to keep the fingers centered (upper links)
            fingerGear = self._p.createConstraint(
                self.panda_id,
                self.pandaLeftFingerJointIndex,
                self.panda_id,
                self.pandaRightFingerJointIndex,
                jointType=self._p.JOINT_GEAR,
                jointAxis=[1, 0, 0],
                parentFramePosition=[0, 0, 0],
                childFramePosition=[0, 0, 0])
            self._p.changeConstraint(fingerGear,
                                     gearRatio=-1,
                                     erp=0.1,
                                     maxForce=2 * self.maxFingerForce)

            # Disable damping for all links
            for i in range(self.numJoints):
                self._p.changeDynamics(self.panda_id,
                                       i,
                                       linearDamping=0,
                                       angularDamping=0)

        # Reset all joint angles
        self.reset_robot_joints(self.init_joint_angles)

    def reset_robot_joints(self, angles):
        """[summary]

        Args:
            angles ([type]): [description]
        """
        if len(angles) < self.numJoints:  # 7
            angles += [
                0, -np.pi / 4, self.fingerOpenPos, 0.00, self.fingerOpenPos,
                0.00
            ]
        for i in range(self.numJoints):  # 13
            self._p.resetJointState(self.pandaId, i, angles[i])

    def reset_arm_joints_ik(self, pos, orn, gripper_closed=False):
        """[summary]

        Args:
            pos ([type]): [description]
            orn ([type]): [description]
            gripper_closed (bool, optional): [description]. Defaults to False.
        """
        jointPoses = list(
            self._p.calculateInverseKinematics(
                self.panda_id,
                self.pandaEndEffectorLinkIndex,
                pos,
                orn,
                jointDamping=self.jd,
                lowerLimits=self.jointLowerLimit,
                upperLimits=self.jointUpperLimit,
                jointRanges=self.jointRange,
                restPoses=self.jointRestPose,
                residualThreshold=1e-4))
        #    , maxNumIterations=1e5))
        if gripper_closed:
            fingerPos = self.fingerClosedPos
        else:
            fingerPos = self.fingerOpenPos
        jointPoses = jointPoses[:7] + [
            0, -np.pi / 4, fingerPos, 0.00, fingerPos, 0.00
        ]
        self.reset_robot_joints(jointPoses)

    # def reset_arm_joints(self, joints):
    #     """[summary]

    #     Args:
    #         joints ([type]): [description]
    #     """
    #     jointPoses = joints + [
    #         0, -np.pi / 4, self._panda.fingerOpenPos, 0.00,
    #         self._panda.fingerOpenPos, 0.00
    #     ]
    #     self._panda.reset(jointPoses)

    @abstractmethod
    def move(self):
        """
        Move robot to the next state; returns next state
        """
        raise NotImplementedError

    def grasp(self, targetVel=0):
        """
        Change gripper velocity direction
        """

        # Use specified velocity if available
        if targetVel > 1e-2 or targetVel < -1e-2:
            self.fingerCurVel = targetVel
        else:
            if self.fingerCurVel > 0.0:
                self.fingerCurVel = -0.05
            else:
                self.fingerCurVel = 0.05
        return

    ################# Get info #################

    def _get_obs(self, camera_params):
        """Return unnormalized depth image

        Args:
            camera_params ([type]): [description]

        Returns:
            [type]: [description]
        """

        viewMat = camera_params['viewMatPanda']
        projMat = camera_params['projMatPanda']
        width_orig = camera_params['width_orig']
        height_orig = camera_params['height_orig']
        near = camera_params['near']
        far = camera_params['far']
        width = camera_params['width']
        height = camera_params['height']

        img_arr = self._p.getCameraImage(width=width_orig,
                                         height=height_orig,
                                         viewMatrix=viewMat,
                                         projectionMatrix=projMat,
                                         flags=self._p.ER_NO_SEGMENTATION_MASK)
        center = width_orig // 2  # assume square for now

        depth = np.reshape(
            img_arr[3],
            (width_orig, height_orig))[center - width // 2:center + width // 2,
                                       center - height // 2:center +
                                       height // 2]
        depth = far * near / (far - (far - near) * depth)
        # depth = (0.3 -
        #          depth) / self.max_obj_height  # set table zero, and normalize
        # depth = depth.clip(min=0., max=1.)
        return depth

    def get_ee(self):
        info = self._p.getLinkState(self.panda_id,
                                    self.pandaEndEffectorLinkIndex)
        return self.array(info[4]), self.array(info[5])

    def get_arm_joints(self):  # use list
        info = self._p.getJointStates(self.panda_id, [0, 1, 2, 3, 4, 5, 6])
        angles = [x[0] for x in info]
        return angles

    def get_gripper_joint(self):
        info = self._p.getJointState(self.panda_id,
                                     self.pandaLeftFingerJointIndex)
        return info[0], info[1]

    def get_ee(self):
        info = self._p.getLinkState(self.panda_id,
                                    self.pandaEndEffectorLinkIndex)
        return np.array(info[4]), np.array(info[5])

    def get_left_finger(self):
        info = self._p.getLinkState(self.panda_id,
                                    self.pandaLeftFingerLinkIndex)
        return np.array(info[4]), np.array(info[5])

    def get_right_finger(self):
        info = self._p.getLinkState(self.panda_id,
                                    self.pandaRightFingerLinkIndex)
        return np.array(info[4]), np.array(info[5])
