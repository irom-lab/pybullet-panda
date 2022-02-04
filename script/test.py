# Minimal working example
import numpy as np
from panda_gym.grasp_env import GraspEnv

# Configure camera
camera_height = 0.40
camera_params = {}
camera_params['pos'] = np.array([0.5, 0.0, camera_height])
camera_params['quat'] = np.array([1.0, 0.0, 0.0, 0.0])  # pointing downwards
camera_params['img_w'] = 128
camera_params['img_h'] = 128
camera_params['aspect'] = 1
camera_params['camera_fov'] = 70    # vertical fov in degrees
camera_params['max_depth'] = camera_height

# Initialize environment
env = GraspEnv(task=None,
                renders=True,
                img_h=128,
                img_w=128,
                use_rgb=False,
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
obs, reward, _, _ = env.step(action=[0.5, 0.0, 0.2])
print('\nReward: {}\n'.format(reward))
