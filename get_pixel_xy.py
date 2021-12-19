import pybullet as p
from panda.panda_env import PandaEnv
import numpy as np

from src.util_depth import pixelToWorld

# Initialize env
p.connect(p.GUI)
panda_env = PandaEnv(mu=0.3, sigma=0.01, finger_type='long')
panda_env.reset_env()

# Load mug
obj_id = p.loadURDF('data/sample_mug/4.urdf',
                    basePosition=[0.5, 0.0, 0.0],
                    baseOrientation=[0, 0, 0, 1])

# Get camera params
width = 64
height = 64
# params['imgW_orig'] = 1024
# params['imgH_orig'] = 768
cam_height = 0.3

# Correct aspect ratio
viewMat = [
    -1.0, 0.0, -0.0, 0.0, -0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.5, 0.0,
    -cam_height, 1.0
]  # 4x4, row first, then column
projMat = [
    1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0000200271606445,
    -1.0, 0.0, 0.0, -0.02000020071864128, 0.0
]
# params['cameraUp'] = [0.0, 0.0, 1.0]
# params['camForward'] = [0.0, -0.00017464162374380976, -1.0]
# params['horizon'] = [-20000.0, -0.0, 0.0]
# params['vertical'] = [0.0, -20000.0, 3.4928321838378906]
# params['dist'] = 0.30
# params['camTarget'] = [0.5, 0.0, 0.0]

# p.resetDebugVisualizerCamera(0.10, 180, -89.99, [0.50, 0.0, 0.0])
# width, height, viewMat, projMat, cameraUp, camForward, horizon, vertical, _, _, dist, camTarget = p.getDebugVisualizerCamera(
# )
# m = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[0.50, 0.0, 0.0],
#                                         distance=0.10,
#                                         yaw=180,
#                                         pitch=-90.0,
#                                         roll=0,
#                                         upAxisIndex=2)
# print(m)
# print(width)
# print(height)
# print(viewMat)
# print(projMat)
# print(cameraUp)
# print(camForward)
# print(horizon)
# print(vertical)
# print(dist)
# print(camTarget)
# # params['viewMatPanda'] = m  # 4x4, row first, then column
# params['viewMatPanda'] = viewMat  # 4x4, row first, then column
# params['projMatPanda'] = projMat
# params['cameraUp'] = cameraUp
# params['camForward'] = camForward
# params['horizon'] = horizon
# params['vertical'] = vertical
# params['dist'] = dist
# params['camTarget'] = camTarget

###########################################################################

# calculations of near and far based on projection matrix
# https://answers.unity.com/questions/1359718/what-do-the-values-in-the-matrix4x4-for-cameraproj.html
# https://forums.structure.io/t/near-far-value-from-projection-matrix/3757
m22 = projMat[10]
m32 = projMat[
    14]  # THe projection matrix (array[15]) returned by PyBullet orders using column first
# params['near'] = 0.01
# params['far'] = 1000
near = 2 * m32 / (2 * m22 - 2)
far = ((m22 - 1.0) * near) / (m22 + 1.0)

params = {}
params['viewMat'] = viewMat
params['projMat'] = projMat
params['near'] = near
params['far'] = far
params['imgW'] = width
params['imgH'] = height
params['cameraUp'] = [0.0, 0.0, 1.0]
params['camForward'] = [0.0, -0.00017464162374380976, -1.0]
# params['camForward'] = [0.0, 0.0, -1.0]
params['horizon'] = [-20000.0, -0.0, 0.0]
params['vertical'] = [0.0, -20000.0, 3.4928321838378906]
# params['vertical'] = [0.0, -20000.0, 0.01]
params['dist'] = 0.1
params['camTarget'] = [0.5, 0.0, 0.0]

# Get depth
img_arr = p.getCameraImage(width=width,
                           height=height,
                           viewMatrix=viewMat,
                           projectionMatrix=projMat,
                           flags=p.ER_NO_SEGMENTATION_MASK)
orig_dim = width
center = orig_dim // 2
crop_dim = 20  # 128: 15cm square; 96: 9cm square

depth = np.reshape(img_arr[3], (width, height))\
                [center - crop_dim // 2:center + crop_dim // 2,
                center - crop_dim // 2:center + crop_dim // 2]
depth = far * near / (far - (far - near) * depth)
# depth = (0.3 - depth) / self.max_obj_height  # set table zero, and normalize
# depth = depth.clip(min=0., max=1.)

pcl = pixelToWorld(img_arr[3],
                   center=orig_dim // 2,
                   dim=crop_dim,
                   params=params)
print(np.min(pcl[:, :, 0]), np.min(pcl[:, :, 1]))
print(np.max(pcl[:, :, 0]), np.max(pcl[:, :, 1]))

import matplotlib.pyplot as plt

plt.imshow(depth, cmap='Greys')
plt.show()
