# Minimal working example
import numpy as np
from panda_gym.grasp_env import GraspEnv
from alano.geometry.transform import euler2quat
import pybullet as p

# Configure camera
camera_height = 0.60
camera_params = {}
camera_params['pos'] = np.array([0.5, 0.8, camera_height])
camera_params['quat'] = p.getQuaternionFromEuler([0, -2*np.pi/3, np.pi/2]) # extrinsic
# camera_params['quat'] = euler2quat([np.pi/2, -4*np.pi/5, 0])  # intrinsic
# camera_params['quat'] = np.array([1.0, 0.0, 0.0, 0.0])  # pointing downwards
camera_params['img_w'] = 640
camera_params['img_h'] = 480
camera_params['aspect'] = 1
camera_params['camera_fov'] = 70    # vertical fov in degrees
camera_params['max_depth'] = camera_height

# Initialize environment
env = GraspEnv(task=None,
                renders=True,
                img_h=128,
                img_w=128,
                use_rgb=True,
                use_depth=True,
                max_steps_train=100,
                max_steps_eval=100,
                done_type='fail',
                #
                mu=0.5,
                sigma=0.03,
                camera_params=camera_params)
env.reset()

# Execute open-loop grasp
obs, reward, _, _ = env.step(action=[0.5, 0.05, 0.2, np.pi/6])
print('\nReward: {}\n'.format(reward))

import matplotlib.pyplot as plt
plt.imshow(np.transpose(obs[1:], (1,2,0)))
plt.show()
