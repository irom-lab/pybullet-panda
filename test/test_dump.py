# Minimal working example of grasping a box
import numpy as np
import time
import pybullet as p

from panda.panda_env import plot_frame_pb
from panda.flip_env import FlipEnv


print()
# Connect to PyBullet in GUI
p.connect(p.GUI, options="--width=2600 --height=1800")
p.resetDebugVisualizerCamera(0.7, 160, -30, [0.5, 0, 0])
# p.setTimeStep(0.001)

# Initialize Panda environment
env = FlipEnv(
    mu=0.5,  # tangential friction coefficient
    sigma=0.1,  # torsional friction coefficient
    finger_type='long'
)
env.reset_env()

# Load an object - assume [x=0.5, y=0] is the center of workspace
spatula_path = 'data/private/spatula/spatula.urdf'
pancake_path = 'data/private/pancake/pancake.urdf'
bowl_path = '/home/allen/data/processed_objects/YCB_simple/21.urdf'
spatula_id = p.loadURDF(
    spatula_path, basePosition=[0.45, 0.1, 0.001],
    baseOrientation=p.getQuaternionFromEuler([0, 0, 0])
)
pancake_id = p.loadURDF(
    pancake_path, basePosition=[0.6, 0.1, 0.005],
    baseOrientation=p.getQuaternionFromEuler([0, 0, 0])
)
bowl_id = p.loadURDF(
    bowl_path, basePosition=[0.90, 0.1, 0.005],
    baseOrientation=p.getQuaternionFromEuler([0, 0, 0])
)
p.changeDynamics(
    pancake_id,
    -1,
    lateralFriction=0.4,
    spinningFriction=0.01,
    rollingFriction=0.001,  #!
    collisionMargin=0.00
)
p.changeDynamics(
    spatula_id, -1, lateralFriction=0.3, spinningFriction=0.01,
    collisionMargin=0.00
)
# p.changeDynamics(
#     env._pandaId,
#     env._panda.pandaLeftFingerLinkIndex,
#     #  lateralFriction=1.0,
#     #  spinningFriction=0.1,
#     collisionMargin=0.00)  # does not work though somehow
# p.changeDynamics(
#     env._pandaId,
#     env._panda.pandaRightFingerLinkIndex,
#     #  lateralFriction=1.0,
#     #  spinningFriction=0.1,
#     collisionMargin=0.00)
for _ in range(100):
    p.stepSimulation()
# print(p.getDynamicsInfo(env._pandaId, env._panda.pandaRightFingerLinkIndex))
print(p.getDynamicsInfo(pancake_id, -1))
# while 1:
#     continue

# Move to above the handle
env.move_pos(
    [0.30, 0.1, 0.25],
    absolute_global_euler=[-np.pi, np.pi,
                           0],  # use yaw-pitch-roll; see utils_geom.py
    numSteps=300
)
# time.sleep(1)

# Move down
env.move_pos([0.30, 0.1, 0.20], absolute_global_euler=[-np.pi, np.pi, 0],
             numSteps=300)
# time.sleep(1)

# Grasp - this only sets the gripper velocity, no simulation step; thus need to call move_pos again to the same position to allow finger closing
env.grasp(targetVel=-0.10)
env.move_pos([0.30, 0.1, 0.195], absolute_global_euler=[-np.pi, np.pi, 0],
             numSteps=100)
# time.sleep(1)

# Get contact on spatula
# contacts = p.getContactPoints(spatula_id,
#                               env._tableId,
#                               linkIndexA=0,
#                               linkIndexB=-1)

# tilt
env.move_pos([0.42, 0.1, 0.195],
             absolute_global_euler=[-np.pi, np.pi + np.pi / 8,
                                    0], numSteps=500)

# Move
env.move_pos([0.52, 0.1, 0.195],
             absolute_global_euler=[-np.pi, np.pi + np.pi / 8,
                                    0], numSteps=100)
time.sleep(0.5)
env.move_pos([0.52, 0.1, 0.21], absolute_global_euler=[-np.pi, np.pi, 0],
             numSteps=200)
time.sleep(0.5)
env.move_pos([0.48, 0.1, 0.21],
             absolute_global_euler=[-np.pi, np.pi + np.pi / 10,
                                    0], numSteps=200)
time.sleep(1)

# Lift fast!
env.velocity_control([0.0, 0.0, 0.0], [0, 0.1, 0], numSteps=50,
                     objId=pancake_id)
env.velocity_control([0.8, 0.0, 0.5], [0, -1.0, 0], numSteps=50,
                     objId=pancake_id)
env.velocity_control([-0.3, 0.0, -0.3], [0.0, 0, 0], numSteps=50,
                     objId=pancake_id)

# Keep pose
env.velocity_control([0, 0.0, 0.0], [0.0, 0, 0], numSteps=3000,
                     objId=pancake_id)
while 1:
    continue
