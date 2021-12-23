from abc import ABC
import numpy as np
import pybullet_data

from panda_gym.base_env import BaseEnv
from panda.util_geom import quatMult, euler2quat, traj_time_scaling, euler2quat, quat2rot, log_rot, full_jacob_pb


class GraspEnv(BaseEnv, ABC):
    def __init__(
        self,
        task=None,
        render=False,
        img_H=128,
        img_W=128,
        use_rgb=False,
        max_steps_train=100,
        max_steps_eval=100,
        done_type='fail',
        #
        mu=0.5,
        sigma=0.01,
        camera_params=None,
    ):
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
        super(BasePandaEnv, self).__init__(
            task=task,
            render=render,
            img_H=img_H,
            img_W=img_W,
            use_rgb=use_rgb,
            max_steps_train=max_steps_train,
            max_steps_eval=max_steps_eval,
            done_type=done_type,
        )
        self.mu = mu
        self.sigma = sigma

        # Object id
        self.obj_id_list = []
        self._urdfRoot = pybullet_data.getDataPath()

        # Camera info
        self.camera_params = camera_params

    @property
    def action_dim(self):
        """
        Dimension of robot action - x,y,yaw
        """
        return 3

    # @abstractmethod
    # def report(self):
    #     """
    #     Print information of robot dynamics and observation.
    #     """
    #     raise NotImplementedError

    # @abstractmethod
    # def visualize(self):
    #     """
    #     Visualize trajectories and value functions.
    #     """
    #     raise NotImplementedError

    def reset_task(self, task):
        """
        Reset the task for the environment. Load object - task
        """

        # Reset obj info
        self.obj_id_list = []
        self.obj_initial_height_list = {}

        # Load all
        obj_path_list = task['obj_path_list']
        obj_init_state_all = task['obj_init_state_all']
        for obj_path, obj_init_state in zip(obj_path_list, obj_init_state_all):
            # pos = [
            #     obj_x_initial[obj_ind], obj_y_initial[obj_ind],
            #     obj_height_list[obj_ind] / 2 + 0.001
            # ]
            obj_id = self._p.loadURDF(
                obj_path,
                basePosition=obj_init_state[:-1],
                baseOrientation=self._p.getQuaternionFromEuler(
                    obj_init_state[-1]))
            self.obj_id_list += [obj_id]

            # Infer number of links - change dynamics for each
            num_joint = self._p.getNumJoints(obj_id)
            link_all = [-1] + [*range(num_joint)]
            for link_id in link_all:
                self._p.changeDynamics(
                    obj_id,
                    link_id,
                    lateralFriction=self.mu,
                    spinningFriction=self.sigma,
                    frictionAnchor=1,
                )

        # Let objects settle (actually do not need since we know the height of object and can make sure it spawns very close to table level)
        for _ in range(10):
            self._p.stepSimulation()

        # Record object initial height (for comparing with final height when checking if lifted). Note that obj_initial_height_list is a dict
        for obj_id in self.obj_id_list:
            pos, _ = self._p.getBasePositionAndOrientation(obj_id)
            self.obj_initial_height_list[obj_id] = pos[2]

    def reset(self, task=None):
        """
        Reset the environment, including robot state, task, and obstacles.
        Initialize pybullet client if 1st time.
        """
        if self._physics_client_id < 0:

            # Initialize PyBullet instance
            self.init_pb()

            # Load table
            self.plane_id = self._p.loadURDF(self._urdfRoot + '/plane.urdf',
                                             basePosition=[0, 0, -1],
                                             useFixedBase=1)
            self.table_id = self._p.loadURDF(
                self._urdfRoot + '/table/table.urdf',
                basePosition=[0.400, 0.000, -0.630 + 0.005],
                baseOrientation=[0., 0., 0., 1.0],
                useFixedBase=1)

            # Set friction coefficient for table
            self._p.changeDynamics(
                self.table_id,
                -1,
                lateralFriction=self.mu,
                spinningFriction=self.sigma,
                frictionAnchor=1,
            )

        # Load arm, no need to settle (joint angle set instantly)
        self.reset_robot(self.mu, self.sigma)

        # Reset task
        self.reset_task(task)

    def step(self, action):
        """
        Gym style step function. Apply action, move robot, get observation,
        calculate reward, check if done.
        
        Assume action in [x,y,yaw]
        """

        # Set arm to starting pose
        initial_ee_pos_before_img = np.array([0.3, -0.5, 0.25])
        initial_ee_orn = np.array([1.0, 0.0, 0.0, 0.0])  # straight down
        self.reset_arm_joints_ik(initial_ee_pos_before_img, initial_ee_orn)
        self.grasp(targetVel=0.10)  # open gripper

        # Execute, reset ik on top of object, reach down, grasp, lift, check success
        ee_pos = action
        ee_pos_before = action + np.array([0, 0, 0.10])
        ee_pos_after = action + np.array([0, 0, 0.05])
        ee_orn = quatMult(euler2quat([action[-1], 0., 0.]), initial_ee_orn)
        for _ in range(3):
            self.reset_arm_joints_ik(ee_pos_before, ee_orn)
            self._p.stepSimulation()
        self.move(ee_pos, absolute_global_quat=ee_orn, numSteps=300)
        self.grasp(targetVel=-0.10)  # always close gripper
        self.move(ee_pos, absolute_global_quat=ee_orn,
                  numSteps=100)  # keep pose until gripper closes
        self.move(ee_pos_after, absolute_global_quat=ee_orn,
                  numSteps=150)  # lift

        # Check if all objects removed
        self.clear_obj()
        if len(self.obj_id_list) == 0:
            reward = 1
        else:
            reward = 0
        return self._get_obs(self.camera_params), reward, True, {}

    def clear_obj(self):
        height = []
        obj_to_be_removed = []
        for obj_id in self.obj_id_list:
            pos, _ = self._p.getBasePositionAndOrientation(obj_id)
            height += [pos[2]]
            if pos[2] - self.obj_initial_height_list[obj_id] > 0.03:
                obj_to_be_removed += [obj_id]

        for obj_id in obj_to_be_removed:
            self._p.removeBody(obj_id)
            self.obj_id_list.remove(obj_id)

    def move(
        self,
        absolute_pos=None,
        relative_pos=None,
        absolute_global_euler=None,  # preferred
        relative_global_euler=None,  # preferred
        relative_local_euler=None,  # not using
        absolute_global_quat=None,  # preferred
        relative_azi=None,  # for arm
        numSteps=50,
        maxJointVel=0.20,
        # timeStep=0,
        # checkContact=False,
        # objId=None,
        posGain=20,
        velGain=5,
        # relative_quat=None,  # never use relative quat
    ):

        # Get trajectory
        eePosNow, eeQuatNow = self.get_ee()

        # Determine target pos
        if absolute_pos is not None:
            targetPos = absolute_pos
        elif relative_pos is not None:
            targetPos = eePosNow + relative_pos
        else:
            targetPos = eePosNow

        # Determine target orn
        if absolute_global_euler is not None:
            targetOrn = euler2quat(absolute_global_euler)
        elif relative_global_euler is not None:
            targetOrn = quatMult(euler2quat(relative_global_euler), eeQuatNow)
        elif relative_local_euler is not None:
            targetOrn = quatMult(eeQuatNow, euler2quat(relative_local_euler))
        elif absolute_global_quat is not None:
            targetOrn = absolute_global_quat
        elif relative_azi is not None:
            # Extrinsic yaw
            targetOrn = quatMult(euler2quat([relative_azi[0], 0, 0]),
                                 eeQuatNow)
            # Intrinsic pitch
            targetOrn = quatMult(targetOrn, euler2quat([0, relative_azi[1],
                                                        0]))
        # elif relative_quat is not None:
        # 	targetOrn = quatMult(eeQuatNow, relative_quat)
        else:
            targetOrn = np.array([1.0, 0., 0., 0.])

        # Get trajectory
        trajPos = traj_time_scaling(startPos=eePosNow,
                                    endPos=targetPos,
                                    numSteps=numSteps)

        # Run steps
        numSteps = len(trajPos)
        for step in range(numSteps):

            # Get joint velocities from error tracking control, takes 0.2ms
            jointDot = self.traj_tracking_vel(targetPos=trajPos[step],
                                              targetQuat=targetOrn,
                                              posGain=posGain,
                                              velGain=velGain)

            # Send velocity commands to joints
            for i in range(self.numJointsArm):
                self._p.setJointMotorControl2(self.panda_id,
                                              i,
                                              self._p.VELOCITY_CONTROL,
                                              targetVelocity=jointDot[i],
                                              force=self.maxJointForce[i],
                                              maxVelocity=maxJointVel)

            # Keep gripper current velocity
            self._p.setJointMotorControl2(self.panda_id,
                                          self.pandaLeftFingerJointIndex,
                                          self._p.VELOCITY_CONTROL,
                                          targetVelocity=self.fingerCurVel,
                                          force=self.maxFingerForce,
                                          maxVelocity=0.10)
            self._p.setJointMotorControl2(self.panda_id,
                                          self.pandaRightFingerJointIndex,
                                          self._p.VELOCITY_CONTROL,
                                          targetVelocity=self.fingerCurVel,
                                          force=self.maxFingerForce,
                                          maxVelocity=0.10)

            # # Quit if contact at either finger
            # if checkContact:
            #     contact = self.check_contact(objId, both=False)
            #     if contact:
            #         return timeStep, False

            # Step simulation, takes 1.5ms
            self._p.stepSimulation()
            # timeStep += 1
        # return timeStep, True

    def traj_tracking_vel(self,
                          targetPos,
                          targetQuat,
                          posGain=20,
                          velGain=5):  #Change gains based off mouse
        eePos, eeQuat = self.get_ee()

        eePosError = targetPos - eePos
        # eeOrnError = log_rot(quat2rot(targetQuat)@(quat2rot(eeQuat).T))  # in spatial frame
        eeOrnError = log_rot(quat2rot(targetQuat).dot(
            (quat2rot(eeQuat).T)))  # in spatial frame

        jointPoses = self.get_arm_joints() + [0, 0, 0]  # add fingers
        eeState = self._p.getLinkState(self.panda_id,
                                       self.pandaEndEffectorLinkIndex,
                                       computeLinkVelocity=1,
                                       computeForwardKinematics=1)
        # Get the Jacobians for the CoM of the end-effector link. Note that in this example com_rot = identity, and we would need to use com_rot.T * com_trn. The localPosition is always defined in terms of the link frame coordinates.
        zero_vec = [0.0] * len(jointPoses)
        jac_t, jac_r = self._p.calculateJacobian(
            self.panda_id, self.pandaEndEffectorLinkIndex, eeState[2],
            jointPoses, zero_vec,
            zero_vec)  # use localInertialFrameOrientation
        jac_sp = full_jacob_pb(
            jac_t, jac_r)[:, :7]  # 6x10 -> 6x7, ignore last three columns

        try:
            jointDot = np.linalg.pinv(jac_sp).dot((np.hstack(
                (posGain * eePosError,
                 velGain * eeOrnError)).reshape(6, 1)))  # pseudo-inverse
        except np.linalg.LinAlgError:
            jointDot = np.zeros((7, 1))

        return jointDot
