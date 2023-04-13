# Minimal working example
import numpy as np
from omegaconf import OmegaConf
from panda_gym.grasp_env import GraspEnv

# Configure camera
camera_height = 5  # orthographic
# camera_height = 0.4
camera_param = OmegaConf.create()
camera_param.pos = [0.5, 0, camera_height]
camera_param.euler = [0, -np.pi, np.pi / 2]  # extrinsic - top-down
camera_param.img_w = 100
camera_param.img_h = 100
camera_param.aspect = 1
camera_param.fov = 5  # orthographic
# camera_param.fov = 60    # vertical fov in degrees
camera_param.use_rgb = True
camera_param.use_depth = True
camera_param.max_depth = camera_height
camera_param.min_depth = camera_height - 0.15

# Resulting focal length and workspace half dimension, assume aspect=1 for now
focal_len = camera_param.img_h / (
    2 * np.tan(camera_param.fov * np.pi / 180 / 2)
)
print('Focal length: ', focal_len)
workspace_half_dim = np.tan(
    camera_param['fov'] / 2 * np.pi / 180
) * camera_height
print('Workspace half dimension: ', workspace_half_dim)

#
task = OmegaConf.create()
# task.obj_path = 'data/sample/42_gavel/42_gavel.urdf'
task.obj_path = 'data/sample/mug/3.urdf'
task.obj_pos = [0.6, 0.1, 0.05]
task.obj_quat = [0, 0, 0, 1]
task.global_scaling = 1.2
task.table_rgba = [0.3, 0.3, 0.3, 1]

# Initialize environment
env = GraspEnv(
    task=None,
    render=True,
    camera_param=camera_param,
    #
    mu=0.5,
    sigma=0.03
)
obs = env.reset(task)

# Execute open-loop grasp
_, reward, _, _ = env.step(action=[-0.05, 0.02, 0.05, np.pi / 2])
print('\nReward: {}\n'.format(reward))

import matplotlib.pyplot as plt
# make subplot
fig, ax = plt.subplots(1, 2)
# display the image
ax[0].imshow(obs[0])
ax[1].imshow(np.transpose(obs[1:] / 255, (1, 2, 0)))
print(np.max(obs[0]))
plt.show()
