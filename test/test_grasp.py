# Minimal working example
import numpy as np
from panda_gym.grasp_env import GraspEnv

# Configure camera
camera_height = 0.60
camera_param = {}
camera_param['pos'] = np.array([0.5, 0, camera_height])
camera_param['euler'] = [0, -np.pi, np.pi/2] # extrinsic - top-down
camera_param['img_w'] = 640
camera_param['img_h'] = 480
camera_param['aspect'] = 1
camera_param['fov'] = 70    # vertical fov in degrees
camera_param['use_rgb'] = False
camera_param['use_depth'] = True
camera_param['overhead_max_depth'] = camera_height
camera_param['overhead_min_depth'] = 0.3
camera_param['save_byte'] = False

# Initialize environment
env = GraspEnv(task=None,
               render=True,
               camera_param=camera_param,
               #
               mu=0.5,
               sigma=0.03)
obs = env.reset()

# Execute open-loop grasp
_, reward, _, _ = env.step(action=[0.49, 0.04, 0.2, np.pi/6])
print('\nReward: {}\n'.format(reward))

import matplotlib.pyplot as plt
plt.imshow(obs[0])
plt.show()
# plt.imshow(np.transpose(obs[1:], (1,2,0)))
# plt.show()