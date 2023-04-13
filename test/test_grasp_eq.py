# Minimal working example
import numpy as np
from panda_gym.grasp_env import GraspEnv
from panda_gym.grasp_flip_env import GraspFlipEnv
import pickle
from omegaconf import OmegaConf
# from util.depth import

# Configure camera
camera_height = 0.4
camera_param = OmegaConf.create()
camera_param.pos = [0.5, 0, camera_height]
camera_param.euler = [0, -np.pi, np.pi / 2]  # extrinsic - top-down
camera_param.img_w = 64
camera_param.img_h = 64
camera_param.aspect = 1
camera_param.fov = 60  # vertical fov in degrees
camera_param.use_rgb = True
camera_param.use_depth = True
camera_param.max_depth = camera_height
camera_param.min_depth = 0.25
# img_half_dim_in_world = np.tan(np.deg2rad(camera_param.fov/2)) * camera_height
# ignore projection for now

# load dataset
dataset = 'data/mug/eq_0'
print("= loading tasks from {}".format(dataset))
with open(dataset + '.pkl', 'rb') as f:
    task_all = pickle.load(f)

# Initialize environment
env = GraspFlipEnv(
    task=None,
    # env = GraspEnv(task=None,
    render=True,
    camera_param=camera_param,
    #
    mu=0.5,
    sigma=0.03
)

for task_ind in range(10):
    task = task_all[task_ind]
    obs = env.reset(task)

    # Execute open-loop grasp
    _, reward, _, _ = env.step(action=[0.49, 0.04, 0.2, np.pi / 6])
    print('\nReward: {}\n'.format(reward))

    depth = obs[0]
    rgb = np.transpose(
        obs[1:] / 255, (1, 2, 0)
    )  # from uint8 (C,H,W) to float [0,1] (H,W,C)

    import matplotlib.pyplot as plt
    # make subplot
    fig, ax = plt.subplots(1, 2)
    # flip the image from left to right
    # display the image
    ax[0].imshow(depth)
    ax[1].imshow(rgb)
    # label each image
    ax[0].set_title('Depth')
    ax[1].set_title('RGB')
    plt.show()
