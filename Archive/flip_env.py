import pybullet as p
import pybullet_data
import numpy as np
import time
import cvxpy as cp

from panda.panda_env import PandaEnv, full_jacob_pb


class FlipEnv(PandaEnv):
    def __init__(self,
                 urdfRoot=pybullet_data.getDataPath(),
                 mu=0.4,
                 sigma=0.03,
                 timestep=1. / 240.,
                 finger_type=None):

        super(FlipEnv, self).__init__(urdfRoot=urdfRoot,
                                      mu=mu,
                                      sigma=sigma,
                                      timestep=timestep,
                                      finger_type=finger_type)

    def velocity_control(self,
                         target_lin_vel,
                         target_ang_vel,
                         numSteps,
                         objId=None):

        target_vel = np.hstack((target_lin_vel, target_ang_vel))

        # jointDot = np.linalg.pinv(jac_sp).dot(target_vel.reshape(
        # 6, 1))  # pseudo-inverse
        # print('Joint dot: ', jointDot)

        # while 1:
        #     continue

        for step in range(numSteps):

            jointPoses = self._panda.get_arm_joints() + [0, 0, 0
                                                         ]  # add fingers
            eeState = p.getLinkState(self._pandaId,
                                     self._panda.pandaEndEffectorLinkIndex,
                                     computeLinkVelocity=1,
                                     computeForwardKinematics=1)

            # Get the Jacobians for the CoM of the end-effector link. Note that in this example com_rot = identity, and we would need to use com_rot.T * com_trn. The localPosition is always defined in terms of the link frame coordinates.
            zero_vec = [0.0] * len(jointPoses)
            jac_t, jac_r = p.calculateJacobian(
                self._pandaId, self._panda.pandaEndEffectorLinkIndex,
                eeState[2], jointPoses, zero_vec,
                zero_vec)  # use localInertialFrameOrientation
            jac_sp = full_jacob_pb(
                jac_t, jac_r)[:, :7]  # 6x10 -> 6x7, ignore last three column

            # Solve for jointDot
            jointDot = cp.Variable(7)
            prob = cp.Problem(
                    cp.Minimize(cp.norm2(jac_sp @ jointDot - target_vel)), \
                    [jointDot >= -self._panda.jointMaxVel, \
                    jointDot <= self._panda.jointMaxVel]
                    )
            prob.solve()
            # print(jointDot.value)
            jointDot = jointDot.value

            # Send velocity commands to joints
            for i in range(self._panda.numJointsArm):
                p.setJointMotorControl2(
                    self._pandaId,
                    i,
                    p.VELOCITY_CONTROL,
                    targetVelocity=jointDot[i],
                    force=self._panda.maxJointForce[i],
                    # maxVelocity=maxJointVel,
                )

            # Keep gripper current velocity
            p.setJointMotorControl2(self._pandaId,
                                    self._panda.pandaLeftFingerJointIndex,
                                    p.VELOCITY_CONTROL,
                                    targetVelocity=self._panda.fingerCurVel,
                                    force=self._panda.maxFingerForce,
                                    maxVelocity=0.10)
            p.setJointMotorControl2(self._pandaId,
                                    self._panda.pandaRightFingerJointIndex,
                                    p.VELOCITY_CONTROL,
                                    targetVelocity=self._panda.fingerCurVel,
                                    force=self._panda.maxFingerForce,
                                    maxVelocity=0.10)

            # Step simulation, takes 1.5ms
            p.stepSimulation()
            # time.sleep(0.01)
            # print(
            #     p.getLinkState(self._pandaId,
            #                    self._panda.pandaEndEffectorLinkIndex,
            #                    computeLinkVelocity=1)[6])
            # print(p.getBaseVelocity(objId)[1])
            # print("===")

        return