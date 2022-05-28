import numpy as np
from pathlib import Path 
import os

from .util import sample_uniform
from alano.geometry.transform import euler2quat, quatMult

class Tool():
    def __init__(self,
                 env,
                 color=[1.0, 0.5, 0.3, 1.0],
                #  base_pos=[0.50, 0, 0],
                 ):
        self._env = env
        self._color = color
        # self._base_pos = base_pos


    def load(self, task):
        env = self._env
        sim = self._env._p

        # Sample yaw if specified
        obj_quat = euler2quat(task['obj_euler'])
        if 'obj_yaw_range' in task:
            obj_yaw = sample_uniform(env.rng, task['obj_yaw_range'])
            obj_quat = quatMult(euler2quat([obj_yaw,0,0]), obj_quat)
 
         # Sample pos if specified
        obj_pos_base = np.array(task['obj_pos_base']) + np.array(task['obj_pos_offset'])
        obj_pos_base[-1] += 0.02
        if 'obj_pos_range' in task:
            obj_x = sample_uniform(env.rng, task['obj_pos_range'][0])
            obj_y = sample_uniform(env.rng, task['obj_pos_range'][1])
            obj_pos_base[0] += obj_x
            obj_pos_base[1] += obj_y
 
        # Load urdf
        home_path = str(Path.home())
        obj_path = os.path.join(home_path, task['obj_path'])
        obj_id = sim.loadURDF(
            obj_path,
            basePosition=obj_pos_base,
            baseOrientation=obj_quat, 
            globalScaling=task['obj_scaling'],
            # flags=sim.URDF_MERGE_FIXED_LINKS,
        )
        # print(sim.getDynamicsInfo(obj_id, -1))

        # Change mass
        if 'link_mass' in task:
            for (link, mass) in task['link_mass']:
                sim.changeDynamics(obj_id, link, mass=mass)
                # print('changing link {} with mass {}'.format(link, mass))
                # print(sim.getDynamicsInfo(obj_id, link))

        # Change color
        sim.changeVisualShape(obj_id, -1, rgbaColor=self._color)
        for link_ind in range(sim.getNumJoints(obj_id)):
            sim.changeVisualShape(obj_id, link_ind, rgbaColor=self._color)
            # print('link: ', link_ind)
            # import time
            # time.sleep(1)
            # sim.changeVisualShape(obj_id, link_ind, rgbaColor=[0.7,0.7,0.7,1])

        # Let objects settle (actually do not need since we know the height of object and can make sure it spawns very close to table level)
        for _ in range(50):
            # Send velocity commands to joints
            for i in range(env._num_joint_arm):
                sim.setJointMotorControl2(env._panda_id,
                    i,
                    sim.VELOCITY_CONTROL,
                    targetVelocity=0,
                    force=env._max_joint_force[i],
                    maxVelocity=env._joint_max_vel[i],
                )
            sim.stepSimulation()

        self.tool_id = obj_id   #* be careful after tool removed
        return obj_id


    def get_pose(self):
        pos, quat = self._env._p.getBasePositionAndOrientation(self.tool_id)
        return np.array(pos), np.array(quat)
