import sys
sys.path.insert(0, '../utils')

import pybullet as p
import pybullet_data
import numpy as np
from numpy import array

from utils_geom import *
from panda import Panda


def full_jacob_pb(jac_t, jac_r):
    return np.vstack((jac_t[0], jac_t[1], jac_t[2], jac_r[0], jac_r[1], jac_r[2]))


class PandaEnv():

	def __init__(self,
				 urdfRoot=pybullet_data.getDataPath(),
				 mu=0.4,
				 sigma=0.05,
				 timestep=1./240.,
				 long_finger=False,
				 wide_finger=False
				 ):

		self._urdfRoot = urdfRoot
		self._timeStep = timestep
		
		self._pandaId = None
		self._planeId = None
		self._tableId = None

		self._mu = mu
		self._sigma = sigma

		self.long_finger = long_finger
		self.wide_finger = wide_finger


	def reset_env(self):
		p.resetSimulation()
		p.setPhysicsEngineParameter(enableConeFriction=1)
		p.setTimeStep(self._timeStep)

		# Set gravity
		p.setGravity(0, 0, -9.81)

		# Load plane and table
		# self._planeId = p.loadURDF(self._urdfRoot+'/plane.urdf', basePosition=[0, 0, -1], useFixedBase=1)
		self._tableId = p.loadURDF(self._urdfRoot+'/table/table.urdf', 
                             		basePosition=[0.400, 0.000, -0.630+0.005], 
                               		baseOrientation=[0., 0., 0., 1.0],
                                	useFixedBase=1)

		# Load arm, no need to settle (joint angle set instantly)
		self._panda = Panda(self.long_finger, self.wide_finger)
		self._pandaId = self._panda.load()

		p.changeDynamics(self._pandaId, self._panda.pandaLeftFingerLinkIndex, lateralFriction=self._mu,
			spinningFriction=self._sigma,
			frictionAnchor=1,
			)
		p.changeDynamics(self._pandaId, self._panda.pandaRightFingerLinkIndex, lateralFriction=self._mu,
			spinningFriction=self._sigma,
			frictionAnchor=1,
			)

		p.changeDynamics(self._tableId, -1, 
			lateralFriction=self._mu,
			spinningFriction=self._sigma,
			frictionAnchor=1,
			)

		# Create a constraint to keep the fingers centered (upper links)
		fingerGear = p.createConstraint(
      					self._pandaId, self._panda.pandaLeftFingerJointIndex,
						self._pandaId, self._panda.pandaRightFingerJointIndex,
						jointType=p.JOINT_GEAR,
						jointAxis=[1, 0, 0],
						parentFramePosition=[0, 0, 0],
						childFramePosition=[0, 0, 0])
		p.changeConstraint(fingerGear, gearRatio=-1, 
                     				   erp=0.1, 
                            		   maxForce=2*self._panda.maxFingerForce)

		# Disable damping for all links      
		for i in range(self._panda.numJoints):
			p.changeDynamics(self._pandaId, i, linearDamping=0, 
					  							angularDamping=0)


	def reset_arm_joints_ik(self, pos, orn, gripper_closed=False):
		jointPoses = list(p.calculateInverseKinematics(self._pandaId, 
										self._panda.pandaEndEffectorLinkIndex, 
										pos, orn, 
										jointDamping=self._panda.jd, 	
										lowerLimits=self._panda.jointLowerLimit,
										upperLimits=self._panda.jointUpperLimit,
										jointRanges=self._panda.jointRange,
										restPoses=self._panda.jointRestPose,
										residualThreshold=1e-4))
									#    , maxNumIterations=1e5))
		if gripper_closed:
			fingerPos = self._panda.fingerClosedPos
		else:
			fingerPos = self._panda.fingerOpenPos
		jointPoses = jointPoses[:7] + [0, -np.pi/4, fingerPos, 0.00, fingerPos, 0.00]
		self._panda.reset(jointPoses)


	def reset_arm_joints(self, joints):
		jointPoses = joints + [0, -np.pi/4, self._panda.fingerOpenPos,
				0.00, self._panda.fingerOpenPos, 0.00]
		self._panda.reset(jointPoses)


	########################* Arm control *#######################


	def move_pos(self, absolute_pos=None,
              		   relative_pos=None, 
			  		   absolute_global_euler=None,  # preferred
					   relative_global_euler=None,  # preferred
					   relative_local_euler=None,   # not using
					   absolute_global_quat=None,   # preferred
					   relative_azi=None,  # for arm
					#    relative_quat=None,  # never use relative quat
				   	   numSteps=50, 
					   maxJointVel=0.20, 
					   timeStep=0, 
					   relativePos=True, 
					   checkContact=False, 
					   objId=None,
					   posGain=20,
					   velGain=5):

		# Get trajectory
		eePosNow, eeQuatNow = self._panda.get_ee()

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
			targetOrn = quatMult(euler2quat([relative_azi[0],0,0]), eeQuatNow)
			# Intrinsic pitch
			targetOrn = quatMult(targetOrn, euler2quat([0,relative_azi[1],0]))
		# elif relative_quat is not None:
		# 	targetOrn = quatMult(eeQuatNow, relative_quat)
		else:
			targetOrn = array([1.0, 0., 0., 0.])

		# Get trajectory
		trajPos = self.traj_time_scaling(startPos=eePosNow, 
								   		 endPos=targetPos, 
									  	 numSteps=numSteps)

		# Run steps
		numSteps = len(trajPos)
		for step in range(numSteps):

			# Get joint velocities from error tracking control, takes 0.2ms
			jointDot = self.traj_tracking_vel(targetPos=trajPos[step], targetQuat=targetOrn, posGain=posGain, velGain=velGain)

			# Send velocity commands to joints
			for i in range(self._panda.numJointsArm):
				p.setJointMotorControl2(self._pandaId, 
									i, 
									p.VELOCITY_CONTROL, 
							  		targetVelocity=jointDot[i], 
									force=self._panda.maxJointForce[i], 
								 	maxVelocity=maxJointVel)

			# Keep gripper current velocity
			p.setJointMotorControl2(self._pandaId, 
						   			self._panda.pandaLeftFingerJointIndex, 
							  		p.VELOCITY_CONTROL, 
									targetVelocity=self._panda.fingerCurVel, 
								 	force=self._panda.maxFingerForce, 
									maxVelocity=0.05)
			p.setJointMotorControl2(self._pandaId, 
						   			self._panda.pandaRightFingerJointIndex, 
							  		p.VELOCITY_CONTROL, 
									targetVelocity=self._panda.fingerCurVel, 
								 	force=self._panda.maxFingerForce, 
								  	maxVelocity=0.05)

			# Quit if contact at either finger
			if checkContact:
				contact = self.check_contact(objId, both=False)
				if contact:
					return timeStep, False


			# Step simulation, takes 1.5ms
			p.stepSimulation()
			timeStep += 1

		return timeStep, True


	def grasp(self, targetVel=0):

		"""
		#* Change gripper velocity direction
		"""

		# Use specified velocity if available
		if targetVel > 1e-2 or targetVel < -1e-2:
			self._panda.fingerCurVel = targetVel

		else:
			if self._panda.fingerCurVel > 0.0:
				self._panda.fingerCurVel = -0.05
			else:
				self._panda.fingerCurVel = 0.05
		return


	def traj_time_scaling(self, startPos, endPos, numSteps):
		trajPos = np.zeros((numSteps, 3))
		for step in range(numSteps):
			s = 3 * (1.0 * step / numSteps) ** 2 - 2 * (1.0 * step / numSteps) ** 3
			trajPos[step] = (endPos-startPos)*s+startPos
		return trajPos


	def traj_tracking_vel(self, targetPos, targetQuat, posGain=20, velGain=5): #Change gains based off mouse
		eePos, eeQuat = self._panda.get_ee()

		eePosError = targetPos - eePos
		# eeOrnError = log_rot(quat2rot(targetQuat)@(quat2rot(eeQuat).T))  # in spatial frame
		eeOrnError = log_rot(quat2rot(targetQuat).dot((quat2rot(eeQuat).T)))  # in spatial frame

		jointPoses = self._panda.get_arm_joints() + [0,0,0]  # add fingers
		eeState = p.getLinkState(self._pandaId,
							self._panda.pandaEndEffectorLinkIndex,
							computeLinkVelocity=1,
							computeForwardKinematics=1)
		# Get the Jacobians for the CoM of the end-effector link. Note that in this example com_rot = identity, and we would need to use com_rot.T * com_trn. The localPosition is always defined in terms of the link frame coordinates.
		zero_vec = [0.0] * len(jointPoses)
		jac_t, jac_r = p.calculateJacobian(self._pandaId, 
									 	self._panda.pandaEndEffectorLinkIndex, 
									  	eeState[2], 
									   	jointPoses, 
										zero_vec, 
										zero_vec)  # use localInertialFrameOrientation
		jac_sp = full_jacob_pb(jac_t, jac_r)[:, :7]  # 6x10 -> 6x7, ignore last three columns
		
		try:
			# jointDot = np.linalg.pinv(jac_sp)@(np.hstack((posGain*eePosError, velGain*eeOrnError)).reshape(6,1))  # pseudo-inverse
			jointDot = np.linalg.pinv(jac_sp).dot((np.hstack((posGain*eePosError, velGain*eeOrnError)).reshape(6,1)))  # pseudo-inverse
		except np.linalg.LinAlgError:
			jointDot = np.zeros((7,1))
			
		return jointDot


	###############################* Contact *##################################

	
	def get_contact(self, objId, minForceThres=1e-1):
		left_contacts = p.getContactPoints(self._pandaId, 
							objId, 
							linkIndexA=self._panda.pandaLeftFingerLinkIndex, 
							linkIndexB=-1)
		right_contacts = p.getContactPoints(self._pandaId, 
							objId, 
							linkIndexA=self._panda.pandaRightFingerLinkIndex, 
							linkIndexB=-1)
		left_contacts = [i for i in left_contacts if i[9] > minForceThres]
		right_contacts = [i for i in right_contacts if i[9] > minForceThres]

		return left_contacts, right_contacts


	def get_finger_force(self, objId):
		left_contacts, right_contacts = self.get_contact(objId)

		left_force = np.zeros((3))
		right_force = np.zeros((3))

		for i in left_contacts:
			left_force += i[9]*np.array(i[7])+i[10]*np.array(i[11])+i[12]*np.array(i[13])
		for i in right_contacts:
			right_force += i[9]*np.array(i[7])+i[10]*np.array(i[11])+i[12]*np.array(i[13])

		leftNormalMag = sum([i[9] for i in left_contacts])
		rightNormalMag = sum([i[9] for i in right_contacts])
		numLeftContact = len(left_contacts)
		numRightContact = len(right_contacts)
		# print((leftNormalMag, rightNormalMag, numLeftContact, numRightContact))

		if numLeftContact < 1 or numRightContact < 1:
			# print((numLeftContact, numRightContact))
			return None
		else:
			return left_force, right_force, \
				np.array(left_contacts[0][6]), np.array(right_contacts[0][6]), \
				leftNormalMag, rightNormalMag


	def check_hold_object(self, objId, minForceThres=10.0):

		"""
		#* Check if grasp was successful
		"""
		left_contacts, right_contacts = self.get_contact(objId)

		leftNormalMag = sum([i[9] for i in left_contacts])
		rightNormalMag = sum([i[9] for i in right_contacts])

		return leftNormalMag > minForceThres and rightNormalMag > minForceThres


	def check_contact(self, objId, both=False):

		# Collect contact info between the object and fingers
		leftContacts, rightContacts = self.get_contact(objId)

		if both:
			if len(leftContacts) > 0 and len(rightContacts) > 0:
				return 1
		else:
			if len(leftContacts) > 0 or len(rightContacts) > 0:
				return 1
		return 0


	###############################* Info *##################################


	def get_ee(self):
		return self._panda.get_ee()

	def get_gripper_vel(self):
		return self._panda.fingerCurVel

	def get_gripper_tip_long(self):
		return self._panda.get_gripper_tip_long()

	def get_arm_joints(self):
		return self._panda.get_arm_joints()

	def get_gripper_joint(self):
		return self._panda.get_gripper_joint()

	def get_left_finger(self):
		return self._panda.get_left_finger()

	def get_right_finger(self):
		return self._panda.get_right_finger()
	
	def get_obs(self):
		return self._panda.get_obs()
