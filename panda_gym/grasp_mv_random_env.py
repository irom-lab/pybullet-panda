import numpy as np

from panda_gym.base_env import unnormalize_tanh
from panda_gym.grasp_mv_env import GraspMultiViewEnv
from alano.geometry.transform import quatMult, euler2quat, euler2quat, quat2euler


class GraspMultiViewRandomEnv(GraspMultiViewEnv):
    def __init__(
        self,
        task=None,
        render=False,
        camera_param=None,
        #
        mu=0.5,
        sigma=0.03,
    ):
        super(GraspMultiViewRandomEnv, self).__init__(
            task=task,
            render=render,
            camera_param=camera_param
            mu=mu,
            sigma=sigma,
        )

        # Overrding
        self.action_low = np.array(
                [-0.03, -0.03, -0.05, -30 * np.pi / 180, -15 * np.pi / 180, -15 * np.pi / 180])
        self.action_high = np.array(
                [0.03, 0.03, -0.03, 30 * np.pi / 180, 15 * np.pi / 180, 15 * np.pi / 180])  #! TODO: tune


    @property
    def state_dim(self):
        """
        Dimension of robot state - 6D + gripper
        """
        return 7


    @property
    def action_dim(self):
        """
        Dimension of robot action - 6D
        """
        return 6


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

        # Add bin - make friction very high so mug does not move much after dropping
        obj_collision_id = self._p.createCollisionShape(shapeType=self._p.GEOM_MESH,
            fileName='data/private/bin/bin.obj',
            # meshScale=[0.8,0.8,1.0],
            flags=self._p.GEOM_FORCE_CONCAVE_TRIMESH)
        obj_visual_id = self._p.createVisualShape(shapeType=self._p.GEOM_MESH,
            # meshScale=[0.8,0.8,1.0],
            fileName='data/private/bin/bin.obj', rgbaColor=[0.95,0.95,0.95,1])
        obj_id = self._p.createMultiBody(baseMass=0, baseCollisionShapeIndex=obj_collision_id, baseVisualShapeIndex=obj_visual_id,
            basePosition=[0.5,0,0],
            baseOrientation=self._p.getQuaternionFromEuler(
                    [0, 0, np.pi/2]))
        self._p.changeDynamics(
            obj_id,
            -1,
            lateralFriction=1,
            spinningFriction=0.1,
            rollingFriction=0.01,
            frictionAnchor=1,
        )
        self._obj_id_list += [obj_id]

        # Load all
        obj_path_list = task['obj_path_list']
        obj_state_all = task['obj_state_all']
        obj_rgba_all = task['obj_rgba_all']
        obj_scale_all = task['obj_scale_all']
        obj_height_all = task['obj_height_all']
        for obj_path, obj_state, obj_rgba, obj_scale, obj_height in zip(obj_path_list, obj_state_all, obj_rgba_all, obj_scale_all, obj_height_all):
            # obj_state[3:] = 0
            # obj_path = '/home/allen/data/processed_objects/SNC_v4_mug/0.urdf'
            # obj_path = '/home/allen/data/processed_objects/SNC_v4/03797390/16.urdf'

            # load urdf
            obj_id = self._p.loadURDF(
                obj_path,
                basePosition=obj_state[:3],
                baseOrientation=euler2quat(obj_state[3:]), 
                globalScaling=obj_scale)
            self._obj_id_list += [obj_id]

            # Infer number of links - change dynamics for each
            num_joint = self._p.getNumJoints(obj_id)
            link_all = [-1] + [*range(num_joint)]
            for link_id in link_all:
                self._p.changeDynamics(
                    obj_id,
                    link_id,
                    lateralFriction=self._mu,
                    spinningFriction=self._sigma,
                    frictionAnchor=1,
                )

            # Change color - assume base link
            for link_id in link_all:
                self._p.changeVisualShape(obj_id, link_id, rgbaColor=obj_rgba)

            # Let objects settle (actually do not need since we know the height of object and can make sure it spawns very close to table level)
            for _ in range(100):
                self._p.stepSimulation()
        for _ in range(100):
            self._p.stepSimulation()

        # Record object initial height (for comparing with final height when checking if lifted). Note that obj_initial_height_list is a dict
        for obj_id in self._obj_id_list:
            pos, _ = self._p.getBasePositionAndOrientation(obj_id)
            self._obj_initial_pos_list[obj_id] = pos


    def step(self, action):
        """
        Gym style step function. Apply action, move robot, get observation,
        calculate reward, check if done.
        
        Assume action in [x,y,yaw,grasp]
        """
        # Count time
        self.step_elapsed += 1

        # Extract action
        raw_action = unnormalize_tanh(action, self.action_low, self.action_high)
        delta_x, delta_y, delta_z, delta_yaw, delta_pitch, delta_roll = raw_action
        ee_pos, ee_quat = self._get_ee()
        ee_pos_nxt = ee_pos
        ee_pos_nxt[0] += delta_x
        ee_pos_nxt[1] += delta_y
        ee_pos_nxt[2] += delta_z

        # Move
        ee_quat_nxt = quatMult(
            euler2quat([delta_yaw, delta_pitch, delta_roll]), ee_quat)
        ee_euler_nxt = quat2euler(ee_quat_nxt)
        collision_obj_id_list = None
        self.move_pose(absolute_pos=ee_pos_nxt,
                        absolute_global_quat=ee_quat_nxt,
                        num_steps=150,
                        collision_obj_id_list=collision_obj_id_list)

        # Check if object moved or gripper rolled or pitched, meaning contact happened
        # flag_obj_move = False
        # flag_ee_move = False
        # for obj_id in self._obj_id_list:
        #     pos, _ = self._p.getBasePositionAndOrientation(obj_id)
        #     if math.sqrt((pos[0] - self._obj_initial_pos_list[obj_id][0])**2 + (pos[1] - self._obj_initial_pos_list[obj_id][1])**2) > 0.01:
        #         # print('moved ', obj_id)
        #         flag_obj_move = True
        #         break
        # ee_euler = quat2euler(self._get_ee()[1])
        # ee_euler_diff = np.sum(np.abs(ee_euler_nxt-ee_euler))
        # if ee_euler_diff > np.pi:
        #     ee_euler_diff = abs(ee_euler_diff - 2*np.pi)
        # # print('ee euler diff: ', ee_euler_diff)
        # if ee_euler_diff > 0.1:
        #     flag_ee_move = True
        #     # print('ee moved')
# 
        # Grasp if last step, and then lift
        if self.step_elapsed == self.max_steps:
            self.grasp(target_vel=-0.10)  # close
            self.move_pose(ee_pos_nxt,  # keep for some time
                            absolute_global_quat=ee_quat_nxt,
                            num_steps=150)
            ee_pos_nxt[2] += 0.1
            self.move_pose(ee_pos_nxt,
                            absolute_global_quat=ee_quat_nxt,
                            num_steps=150)
        else:
            self.grasp(target_vel=0.10)  # open

        # Check if all objects removed
        reward = 0.2
        success = 0
        if self.step_elapsed == self.max_steps:
            num_obj_clear = self.clear_obj(thres=0.03)
            if num_obj_clear > 0:
                reward = 1
                success = 1
        # if flag_obj_move or flag_ee_move:   # for last step, this is checking before closing gripper and lifting
        #     reward = 0
        #     self.safe_trial = False

        # Check termination condition
        done = False
        if self.step_elapsed == self.max_steps:
            done = True
        return self._get_obs(), reward, done, {'success': success, 'safe': self.safe_trial}
