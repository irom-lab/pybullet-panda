# Minimal working example
import numpy as np
from omegaconf import OmegaConf
from panda_gym.sweep_env import SweepEnv
from util.geom import euler2quat
import pybullet as p
import time
import pickle


# Configure camera
camera_height = 0.5
camera_param = OmegaConf.create()
camera_param.pos = [0.9, 0, camera_height]
# camera_param.euler = [0, -np.pi, 0] # extrinsic - x up, z forward
camera_param.euler = [0, -3*np.pi/4, 0] # extrinsic - x up, z forward
camera_param.img_w = 128
camera_param.img_h = 128
camera_param.aspect = 1
camera_param.fov = 70    # vertical fov in degrees
camera_param.max_depth = camera_height
camera_param.wrist_offset = [0.05, 0.0, 0.02]

# Dataset
dataset = '/home/allen/meta-lang/data/tool_v5_sweep_train.pkl'
# dataset = '/home/allen/meta-lang/data/tool_v2_test.pkl'
print("Load tasks from", dataset)
with open(dataset, 'rb') as f:
    task_all = pickle.load(f)

# task_all[0]['obj_com_offset'] = [0, -0.15, 0]

# Initialize environment
env = SweepEnv(task=task_all[3],
                render=True,
                use_rgb=True,
                use_depth=False,
                #
                mu=0.5,
                sigma=0.03,
                camera_param=camera_param)
env.seed(0)
env.reset()

# cnt = 0
# while 1:    # needs 19N to counter resistance, and jammed if with additional 10N in x or z
#     import time
#     s1 = time.time()
#     if cnt % 100 < 20:
#         env._p.applyExternalForce(env.peg_id, -1, forceObj=[0,-10,10], 
#                                 posObj=[0.50, -0.10, 0.1], 
#                                 flags=env._p.WORLD_FRAME)
#     env._p.stepSimulation()
#     print(time.time()-s1)
#     cnt += 1


# Execute open-loop grasp
for _ in range(2):
    for step in range(30):
        if step < 10:
            obs, reward, done, info = env.step(action=np.array([0.0, 0.1, -1.0, 
                                                                # 0.0, 0.0,
                                                                0]))
        else:
            obs, reward, done, info = env.step(action=np.array([0.0, 0., 0.1, 
                                                                # 0.0, 1.0, 
                                                                0.0]))

        ee_pos = info['s'][:3]
        print('\nStep: {}, Reward: {:.3f}, Done: {}, x: {:.3f}, y: {:.3f}, z: {:.3f}\n'.format(step, reward, done, ee_pos[0], ee_pos[1], ee_pos[2]))
        time.sleep(0.3)

        import matplotlib.pyplot as plt
        plt.imshow(np.transpose(obs, (1,2,0)))
        plt.show()
    
    # Reset
    # env.reset()
