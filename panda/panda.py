import os
import pybullet as p
import numpy as np
from numpy import array


class Panda:
    def __init__(self, finger_type=None):
        if finger_type is None:
            finger_name = 'panda_arm_finger_orig'
        elif finger_type == 'long':
            finger_name = 'panda_arm_finger_long'
        elif finger_type == 'wide_curved':
            finger_name = 'panda_arm_finger_wide_curved'
        elif finger_type == 'wide_flat':
            finger_name = 'panda_arm_finger_wide_flat'
        else:
            raise NotImplementedError
        self.urdfRootPath = os.path.join(
            os.path.dirname(__file__), f'geometry/franka/{finger_name}.urdf')

        self.numJoints = 13
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

    def load(self):
        self.pandaId = p.loadURDF(
            self.urdfRootPath,
            basePosition=[0, 0, 0],
            baseOrientation=[0, 0, 0, 1],
            useFixedBase=1,
            flags=(p.URDF_USE_SELF_COLLISION
                   and p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT))
        iniJointAngles = [
            0, -0.1, 0, -2, 0, 1.8, 0.785, 0, -np.pi / 4, self.fingerOpenPos,
            0.00, self.fingerOpenPos, 0.00
        ]
        self.reset(iniJointAngles)

        return self.pandaId

    def reset(self, angles):  # use list
        if len(angles) < self.numJoints:  # 7
            angles += [
                0, -np.pi / 4, self.fingerOpenPos, 0.00, self.fingerOpenPos,
                0.00
            ]
        for i in range(self.numJoints):  # 13
            p.resetJointState(self.pandaId, i, angles[i])

    def get_arm_joints(self):  # use list
        info = p.getJointStates(self.pandaId, [0, 1, 2, 3, 4, 5, 6])
        angles = [x[0] for x in info]
        return angles

    def get_gripper_joint(self):
        info = p.getJointState(self.pandaId, self.pandaLeftFingerJointIndex)
        return info[0], info[1]

    def get_ee(self):
        info = p.getLinkState(self.pandaId, self.pandaEndEffectorLinkIndex)
        return array(info[4]), array(info[5])

    def get_left_finger(self):
        info = p.getLinkState(self.pandaId, self.pandaLeftFingerLinkIndex)
        return array(info[4]), array(info[5])

    def get_right_finger(self):
        info = p.getLinkState(self.pandaId, self.pandaRightFingerLinkIndex)
        return array(info[4]), array(info[5])

    def get_obs(self):
        observation = []

        # ee
        state = p.getLinkState(self.pandaId,
                               self.pandaEndEffectorLinkIndex,
                               computeLinkVelocity=1)
        observation += [[
            array(state[0]),
            array(state[1]),
            array(state[6]),
            array(state[7])
        ]]  # pos/orn/velL/velA

        # joints (arm and fingers) pos and vel
        jointStates = p.getJointStates(self.pandaId,
                                       [0, 1, 2, 3, 4, 5, 6, 9, 11])
        jointPoses = [[x[0], x[1]] for x in jointStates]
        observation += [jointPoses]

        return observation
