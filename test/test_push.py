# Minimal working example
import numpy as np
from panda_gym.push_env import PushEnv
from panda_gym.push_tool_env import PushToolEnv
from alano.geometry.transform import euler2quat
import pybullet as p
import time
import pickle

# Configure camera
camera_height = 0.40
camera_params = {}
# camera_params['pos'] = np.array([0.7, 0, camera_height])
camera_params['pos'] = np.array([1.0, 0, camera_height])
# camera_params['euler'] = [0, -np.pi, 0] # extrinsic - x up, z forward
camera_params['euler'] = [0, -3*np.pi/4, 0] # extrinsic - x up, z forward
camera_params['img_w'] = 64
camera_params['img_h'] = 64
camera_params['aspect'] = 1
camera_params['fov'] = 70    # vertical fov in degrees
camera_params['max_depth'] = camera_height

# Dataset
# dataset = 'data/private/box/slim_100_0.pkl'
# dataset = 'data/private/box/slim_100_1.pkl'
# dataset = 'data/private/box/slim_100_2.pkl'
# dataset = '/home/allen/meta-lang/data/tool_v0_train.pkl'
dataset = '/home/allen/meta-lang/data/tool_v0_test.pkl'
print("Load tasks from", dataset)
with open(dataset, 'rb') as f:
    task_all = pickle.load(f)

# task_all[0]['obj_com_offset'] = [0, -0.15, 0]

# Initialize environment
env = PushToolEnv(task=task_all[3],
                    renders=True,
                    use_rgb=True,
                    use_depth=True,
                    #
                    mu=0.3,
                    sigma=0.01,
                    camera_params=camera_params)
env.reset()
# self.reset_arm_joints_ik([0.39, 0.0, 0.17], orn=euler2quat([np.pi,np.pi-np.pi/8,0]))
# while 1:
#     continue

# Execute open-loop grasp
for _ in range(25):
    obs, reward, done, info = env.step(action=np.array([0, 1.0, 1.0]))
    print('\nReward: {}, Done: {}, x: {}, y: {}\n'.format(reward, done, info['ee_pos'][0], info['ee_pos'][1]))
    time.sleep(0.1)

    import matplotlib.pyplot as plt
    plt.imshow(np.transpose(obs[1:], (1,2,0)))
    plt.show()
