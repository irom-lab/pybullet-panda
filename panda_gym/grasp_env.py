import numpy as np
import time
from omegaconf import OmegaConf

from panda_gym.panda_env import PandaEnv
from util.geom import quatMult, euler2quat


class GraspEnv(PandaEnv):
    def __init__(
        self,
        task=None,
        render=False,
        camera_param=None,
        #
        mu=0.5,
        sigma=0.03,
        x_offset=0.5,
        grasp_z_offset=-0.03,
        lift_threshold=0.02,
    ):
        """
        """
        super(GraspEnv, self).__init__(
            task=task,
            render=render,
            camera_param=camera_param,
        )
        self._mu = mu
        self._sigma = sigma
        self._x_offset = x_offset
        self._grasp_z_offset = grasp_z_offset  # grasp happens at the pixel depth plus this offset
        self._lift_threshold = lift_threshold

        # Object id
        self._obj_id_list = []
        self._obj_initial_pos_list = {}

        # Constants
        self.initial_ee_pos_before_img = np.array([0.3, -0.5, 0.25])
        self.initial_ee_orn = np.array([1.0, 0.0, 0.0, 0.0])  # straight down

        # Default task
        if task is None:
            self.task = OmegaConf.create()
            self.task.obj_path = 'data/sample/mug/3.urdf'
            self.task.obj_pos = [0.45, 0.05, 0.15]
            self.task.obj_quat = [0, 0, 0, 1]
            self.task.global_scaling = 1.0


    @property
    def state_dim(self):
        """
        Dimension of robot state - 3D + gripper
        """
        return 4


    @property
    def action_dim(self):
        """
        Dimension of robot action - x, y, z, yaw
        """
        return 4


    @property
    def init_joint_angles(self):
        """
        Initial joint angles for the task - [0.45, 0, 0.40], straight down - ee to finger tip is 15.5cm
        """
        return [
            0, -0.255, 0, -2.437, 0, 2.181, 0.785, 0, 0,
            self._finger_open_pos, 0.00, self._finger_open_pos, 0.00
        ]


    def reset_task(self, task=None):
        """
        Reset the task for the environment. Load object - task
        """
        if task is None:
            task = self.default_task

        # Clean table
        for obj_id in self._obj_id_list:
            self._p.removeBody(obj_id)

        # Reset obj info
        self._obj_id_list = []
        self._obj_initial_pos_list = {}

        # Load urdf
        task.obj_pos[0] = 0.6
        task.obj_pos[1] = 0.05
        task.obj_quat[2] = 0.5
        obj_id = self._p.loadURDF(task.obj_path,
                                  basePosition=task.obj_pos,
                                  baseOrientation=task.obj_quat,
                                  globalScaling=task.global_scaling)
        self._obj_id_list += [obj_id]

        # Let objects settle (actually do not need since we know the height of object and can make sure it spawns very close to table level)
        for _ in range(50):
            self._p.stepSimulation()

        # Record object initial height (for comparing with final height when checking if lifted). Note that obj_initial_height_list is a dict
        for obj_id in self._obj_id_list:
            pos, _ = self._p.getBasePositionAndOrientation(obj_id)
            self._obj_initial_pos_list[obj_id] = pos


    def step(self, action):
        """
        Gym style step function. Apply action, move robot, get observation,
        calculate reward, check if done.
        
        Assume action in [x,y,z,yaw]
        """

        # Set arm to starting pose
        self.reset_arm_joints_ik(self.initial_ee_pos_before_img, self.initial_ee_orn)
        self.grasp(target_vel=0.10)  # open gripper

        # Execute, reset ik on top of object, reach down, grasp, lift, check success
        ee_pos = action[:3]
        ee_pos[0] += self._x_offset
        ee_pos[2] = np.maximum(0, ee_pos[2] + self.ee_finger_offset + self._grasp_z_offset)
        ee_pos_before = ee_pos + np.array([0, 0, 0.10]) 
        ee_pos_after = ee_pos + np.array([0, 0, 0.05])
        ee_orn = quatMult(euler2quat([action[3], 0., 0.]), self.initial_ee_orn)
        for _ in range(3):
            self.reset_arm_joints_ik(ee_pos_before, ee_orn)
            self._p.stepSimulation()
        self.move_pose(ee_pos, absolute_global_quat=ee_orn, num_steps=300)
        self.grasp(target_vel=-0.10)  # always close gripper
        self.move_pose(ee_pos, absolute_global_quat=ee_orn,
                  num_steps=100)  # keep pose until gripper closes
        self.move_pose(ee_pos_after, absolute_global_quat=ee_orn,
                  num_steps=150)  # lift

        # Check if all objects removed
        self.clear_obj()
        if len(self._obj_id_list) == 0:
            reward = 1
        else:
            reward = 0
        return np.array([]), reward, True, {'global_scaling': self.task['global_scaling']}   # s, reward, done, info


    def clear_obj(self, thres=None):
        if thres is None:
            thres = self._lift_threshold
        height = []
        obj_to_be_removed = []
        for obj_id in self._obj_id_list:
            pos, _ = self._p.getBasePositionAndOrientation(obj_id)
            height += [pos[2]]
            if pos[2] - self._obj_initial_pos_list[obj_id][2] > thres:
                obj_to_be_removed += [obj_id]
        for obj_id in obj_to_be_removed:
            self._p.removeBody(obj_id)
            self._obj_id_list.remove(obj_id)
        return len(obj_to_be_removed)


    def _get_obs(self, camera_param):
        return self.get_overhead_obs(camera_param)
