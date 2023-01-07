# Minimal working example
import numpy as np
from panda_gym.push_env import PushEnv
from panda_gym.push_tool_env import PushToolEnv
from util.geom import euler2quat
import pybullet as p
import time
import pickle

# Configure camera
camera_height = 0.40
camera_param = {}
# camera_param['pos'] = np.array([0.7, 0, camera_height])
camera_param['pos'] = np.array([1.0, 0, camera_height])
# camera_param['euler'] = [0, -np.pi, 0] # extrinsic - x up, z forward
camera_param['euler'] = [0, -3*np.pi/4, 0] # extrinsic - x up, z forward
camera_param['img_w'] = 128
camera_param['img_h'] = 128
camera_param['aspect'] = 1
camera_param['fov'] = 60    # vertical fov in degrees
camera_param['overhead_max_depth'] = camera_height
camera_param['overhead_min_depth'] = 0

# Dataset
dataset = '/home/allen/meta-lang/data/tool_v5_push_train.pkl'
print("Load tasks from", dataset)
with open(dataset, 'rb') as f:
    task_all = pickle.load(f)

# task_all[0]['obj_com_offset'] = [0, -0.15, 0]
# task_all[0]['obj_euler'][0] = 2.0
# task_all[0]['obj_euler'][2] = 3.14
# task_all[0]['obj_scaling'] = 2

# Initialize environment
env = PushToolEnv(task=task_all[0],
                    render=True,
                    use_rgb=True,
                    use_depth=True,
                    #
                    mu=0.5,
                    sigma=0.1,
                    camera_param=camera_param)
env.seed(0)
# env.reset()
# self.reset_arm_joints_ik([0.39, 0.0, 0.17], orn=euler2quat([np.pi,np.pi-np.pi/8,0]))
# while 1:
#     continue

# Execute open-loop grasp
for task in task_all:
    env.reset(task)
    for step in range(1):
        obs, reward, done, info = env.step(action=np.array([0.5, 0.3, 0.5]))
        ee_pos = info['s'][:3]
        print('\nStep: {}, Reward: {:.3f}, Done: {}, x: {:.3f}, y: {:.3f}, z: {:.3f}\n'.format(step, reward, done, ee_pos[0], ee_pos[1], ee_pos[2]))
        time.sleep(1)
    # input("Press Enter to continue...")

        import matplotlib.pyplot as plt
        plt.imshow(np.transpose(obs[1:], (1,2,0)))
        plt.show()
    # env.reset()
