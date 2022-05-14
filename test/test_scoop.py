# Minimal working example of grasping a box
import numpy as np
import time
import pybullet as p
import random

from panda.panda_env import plot_frame_pb
from panda.flip_env import FlipEnv

print()
# Connect to PyBullet in GUI
p.connect(p.GUI, options="--width=2600 --height=1800")
p.resetDebugVisualizerCamera(0.6, 150, -25, [0.60, 0, 0])
# p.setTimeStep(0.001)
# p.setRealTimeSimulation(1)

# Initialize Panda environment
env = FlipEnv(
    mu=0.5,  # tangential friction coefficient
    sigma=0.1,  # torsional friction coefficient
    finger_type='long')
env.reset_env()

# Load an object - assume [x=0.5, y=0] is the center of workspace
spatula_path = 'data/private/spatula/spatula.urdf'
# pancake_path = 'data/private/pancake/pancake.urdf'
# bowl_path = '/home/allen/data/processed_objects/YCB_simple/21.urdf'
spatula_id = p.loadURDF(spatula_path,
                        basePosition=[0.45, 0.1, 0.001],
                        baseOrientation=p.getQuaternionFromEuler([0, 0, 0]))
# pancake_id = p.loadURDF(pancake_path,
#                         basePosition=[0.6, 0.1, 0.005],
#                         baseOrientation=p.getQuaternionFromEuler([0, 0, 0]))
for _ in range(10):
    obs_collision_id = p.createCollisionShape(p.GEOM_SPHERE, radius=0.005)
    obs_visual_id = p.createVisualShape(
        p.GEOM_SPHERE,
        radius=0.005,
        rgbaColor=[168 / 255, 214 / 255, 131 / 255, 1])

    obs_id = p.createMultiBody(baseMass=0.002,
                               baseCollisionShapeIndex=obs_collision_id,
                               baseVisualShapeIndex=obs_visual_id,
                               basePosition=[
                                   0.6 + random.random() * 0.003,
                                   0.1 + random.random() * 0.003, 0.005
                               ],
                               baseOrientation=[0, 0, 0, 1])
    p.changeDynamics(
        obs_id,
        -1,
        lateralFriction=0.5,
        spinningFriction=0.1,
        rollingFriction=0.1,  #!
        collisionMargin=0.00)
    for _ in range(10):
        p.stepSimulation()
for _ in range(100):
    p.stepSimulation()

p.changeDynamics(
    spatula_id,
    -1,
    lateralFriction=0.6,
    spinningFriction=0.1,
    rollingFriction=0.01,  #!
    collisionMargin=0.00)
for _ in range(100):
    p.stepSimulation()

# Move to above the handle
env.move_pos(
    [0.30, 0.1, 0.25],
    absolute_global_euler=[-np.pi, np.pi,
                           0],  # use yaw-pitch-roll; see utils_geom.py
    numSteps=1000)
# time.sleep(1)

# Move down
env.move_pos([0.30, 0.1, 0.20],
             absolute_global_euler=[-np.pi, np.pi, 0],
             numSteps=1000)
# time.sleep(1)

# Grasp - this only sets the gripper velocity, no simulation step; thus need to call move_pos again to the same position to allow finger closing
env.grasp(targetVel=-0.10)

# Move
env.move_pos([0.30, 0.1, 0.195],
             absolute_global_euler=[-np.pi, np.pi, 0],
             numSteps=1000)
time.sleep(1)

# tilt
env.move_pos([0.42, 0.1, 0.195],
             absolute_global_euler=[-np.pi, np.pi + np.pi / 20, 0],
             numSteps=500)
# while 1:
#     continue

for i in range(10):
    x_vel = i * 0.1
    z_vel = 0
    if i > 5:
        z_vel = -(i - 5) * 0.1
    env.velocity_control([x_vel, 0.0, 0.0], [0, z_vel, 0],
                         numSteps=5,
                         objId=spatula_id)
    # time.sleep(0.1)

# Curl up
for i in range(10):
    x_vel -= 0.1
    z_vel += 0.005
    env.velocity_control([x_vel, 0.0, 0.01], [0, z_vel, 0],
                         numSteps=5,
                         objId=spatula_id)
    # time.sleep(0.1)

# Level
env.velocity_control([0, 0.0, 0.01], [0, 0.01, 0],
                     numSteps=20,
                     objId=spatula_id)
# time.sleep(0.1)

# Keep pose
env.velocity_control([0, 0.0, 0.005], [0.0, 0, 0],
                     numSteps=2000,
                     objId=spatula_id)
while 1:
    continue
