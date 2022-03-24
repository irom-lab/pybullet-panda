# Minimal working example
import numpy as np
from panda_gym.push_env import PushEnv
from alano.geometry.transform import euler2quat
import pybullet as p
import time
import pickle

# Configure camera
camera_height = 0.60
camera_params = {}
camera_params['pos'] = np.array([0.5, 0, camera_height])
camera_params['euler'] = [0, -np.pi, np.pi/2] # extrinsic
# camera_params['quat'] = euler2quat([np.pi/2, -4*np.pi/5, 0])  # intrinsic
# camera_params['quat'] = np.array([1.0, 0.0, 0.0, 0.0])  # pointing downwards
camera_params['img_w'] = 64
camera_params['img_h'] = 64
camera_params['aspect'] = 1
camera_params['fov'] = 70    # vertical fov in degrees
camera_params['max_depth'] = camera_height

# Dataset
dataset = 'data/private/box/box_100_0.pkl'
print("Load tasks from", dataset)
with open(dataset, 'rb') as f:
    task_all = pickle.load(f)

# Initialize environment
env = PushEnv(task=task_all[0],
            renders=True,
            use_rgb=True,
            use_depth=True,
            #
            mu=0.5,
            sigma=0.03,
            camera_params=camera_params)
env.reset()
# self.reset_arm_joints_ik([0.39, 0.0, 0.17], orn=euler2quat([np.pi,np.pi-np.pi/8,0]))
# while 1:
#     continue

# Execute open-loop grasp
for _ in range(5*5):
    obs, reward, done, info = env.step(action=np.array([0.3, 0, 0]))
    print('\nReward: {}, Done: {}, x: {}, y: {}\n'.format(reward, done, info['ee_pos'][0], info['ee_pos'][1]))
    time.sleep(0.1)

    import matplotlib.pyplot as plt
    plt.imshow(np.transpose(obs[1:], (1,2,0)))
    plt.show()
