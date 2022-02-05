# Custom Franka Panda environment in PyBullet

Some sample environments are provided in ```panda_gym``` that follow the OpenAI Gym environment style.  In ```script```, ```test.py``` provides a minimal working example, and ```run_gym.py``` provides an example of running vectorized environments using [Stable Baselines 3](https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html). 

### Other functionality

1. Differential IK controller that supports a wide variety of orientation input (Euler/Quaternion/Azimuthal)
2. Different gripper finger models: (1) the original one from Franka; (2) a longer one (preferred for grasping, design from [Doug Morrison](https://dougsm.com/)); (3) a wider one (for grasping cups and pouring liquid).
