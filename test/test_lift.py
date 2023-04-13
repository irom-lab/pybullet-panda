# Minimal working example
import numpy as np
from panda_gym.lift_env import LiftEnv
from util.geom import euler2quat
from omegaconf import OmegaConf
import pybullet as p
import time
import pickle

# Configure camera
camera_height = 0.40
camera_param = OmegaConf.create()
camera_param.pos = np.array([0.9, 0, camera_height])
# camera_par.['euler'] = [0, -np.pi, 0] # extrinsic - x up, z forward
camera_param.euler = [0, -3 * np.pi / 4, 0]  # extrinsic - x up, z forward
camera_param.img_w = 128
camera_param.img_h = 128
camera_param.aspect = 1
camera_param.fov = 60  # vertical fov in degrees
camera_param.max_depth = 0.8
camera_param.min_depth = 0.3
camera_param.wrist_offset = [0.05, 0.0, 0.02]
camera_param.wrist_max_depth = 0.4

# Dataset
dataset = '/home/allen/meta-lang/data/tool_v5_lift_train.pkl'
# dataset = '/home/allen/meta-lang/data/tool_v4_lift_test.pkl'
print("Load tasks from", dataset)
with open(dataset, 'rb') as f:
    task_all = pickle.load(f)

# task_all[0]['obj_com_offset'] = [0, -0.15, 0]

# Initialize environment
env = LiftEnv(
    task=task_all[4],
    render=True,
    use_rgb=True,
    use_depth=True,
    #
    mu=0.5,
    sigma=0.03,
    camera_param=camera_param
)
env.seed(0)
env.reset()
# self.reset_arm_joints_ik([0.39, 0.0, 0.17], orn=euler2quat([np.pi,np.pi-np.pi/8,0]))
# while 1:
#     continue

# Execute open-loop grasp
for _ in range(2):
    for step in range(30):
        if step < 20:
            obs, reward, done, info = env.step(
                action=np.array([
                    0.2,
                    0.5,
                    -1,
                    0.0,
                ])
            )
        else:
            obs, reward, done, info = env.step(
                action=np.array([
                    0.0,
                    0.,
                    1,
                    0.0,
                ])
            )
        ee_pos = info['s'][:3]
        print(
            '\nStep: {}, Reward: {:.3f}, Done: {}, x: {:.3f}, y: {:.3f}, z: {:.3f}\n'
            .format(step, reward, done, ee_pos[0], ee_pos[1], ee_pos[2])
        )
        time.sleep(0.3)

        # import matplotlib.pyplot as plt
        # # plt.imshow(np.transpose(obs[3], (1,2,0)))
        # plt.imshow(obs[0])
        # plt.show()

        # Reset
        # env.reset()
