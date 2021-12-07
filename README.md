# Custom Franka Panda environment in PyBullet

```move_pos``` in panda_env.py implements a resolved-rate velocity controller of the arm. A wide variety of orientation input (Euler/Quaternion/Azimuthal) is supported. utils_geom.py specifies the Euler angle convention used.

There are three different fingers available:
1. the original one from Franka
2. a longer one (preferred for grasping, design from [Doug Morrison](https://dougsm.com/))
3. a wider one (for grasping cups and pouring liquid)

train_grasp.py implements simple 2D grasp training using affordance map.
