from utils import *
import math


def getParameters():
    params = {}

    # params['imgW'] = 256
    # params['imgH'] = 256
    params['imgW'] = 512
    params['imgH'] = 512

    params['imgW_orig'] = 1024
    params['imgH_orig'] = 768

    ## Works but aspect ratio wrong
    # params['viewMatPanda'] = [-1.0, 0.0, -0.0, 0.0, -0.0, -1.0, 0.00017464162374380976, 0.0, 0.0, 0.00017464162374380976, 1.0, 0.0, 0.5, -3.637978807091713e-12, -0.30000001192092896, 1.0]  # 4x4, row first, then column
    # params['projMatPanda'] = [0.5472221970558167, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0000200271606445, -1.0, 0.0, 0.0, -0.02000020071864128, 0.0]
    # params['cameraUp'] = [0.0, 0.0, 1.0]
    # params['camForward'] = [0.0, -0.00017464162374380976, -1.0]
    # params['horizon'] = [-36548.22265625, -0.0, 0.0]
    # params['vertical'] = [0.0, -20000.0, 3.4928321838378906]
    # params['dist'] = 0.30000001192092896
    # params['camTarget'] = [0.5, 0.0, 0.0]

    # Correct aspect ratio
    params['viewMatPanda'] = [
        -1.0, 0.0, -0.0, 0.0, -0.0, -1.0, 0.00017464162374380976, 0.0, 0.0,
        0.00017464162374380976, 1.0, 0.0, 0.5, 0.0, -0.300, 1.0
    ]  # 4x4, row first, then column
    params['projMatPanda'] = [
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0000200271606445,
        -1.0, 0.0, 0.0, -0.02000020071864128, 0.0
    ]
    params['cameraUp'] = [0.0, 0.0, 1.0]
    params['camForward'] = [0.0, -0.00017464162374380976, -1.0]
    params['horizon'] = [-20000.0, -0.0, 0.0]
    params['vertical'] = [0.0, -20000.0, 3.4928321838378906]
    params['dist'] = 0.30
    params['camTarget'] = [0.5, 0.0, 0.0]

    # p.resetDebugVisualizerCamera(0.30, 180, -89.99, [0.50, 0.0, 0.0])
    # width, height, viewMat, projMat, cameraUp, camForward, horizon, vertical, _, _, dist, camTarget = p.getDebugVisualizerCamera()
    # # m = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[0.50,0.0,0.0], distance=0.30, yaw=180, pitch=-90.0, roll=0, upAxisIndex=2)
    # # print(m)
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
    m22 = params['projMatPanda'][10]
    m32 = params['projMatPanda'][
        14]  # THe projection matrix (array[15]) returned by PyBullet orders using column first
    # params['near'] = 0.01
    # params['far'] = 1000
    params['near'] = 2 * m32 / (2 * m22 - 2)
    params['far'] = ((m22 - 1.0) * params['near']) / (m22 + 1.0)

    # print('Near', params['near'])
    # print('Far: ', params['far'])

    return params


# Camera pixel location to 3D world location (point cloud)
def pixelToWorld(depthBuffer, center, dim, params=None):
    point_cloud = np.zeros((0, 3))
    if params is None:
        params = getParameters()
    far = params['far']  # 998.6
    near = params['near']  # 0.01

    pixel2xy = np.zeros((dim, dim, 2))
    stepX = 1
    stepY = 1
    for w in range(center - dim // 2, center + dim // 2, stepX):
        for h in range(center - dim // 2, center + dim // 2, stepY):
            rayFrom, rayTo, alpha = getRayFromTo(w, h, params)
            rf = np.array(rayFrom)
            rt = np.array(rayTo)
            vec = rt - rf
            l = np.sqrt(np.dot(vec, vec))
            depthImg = float(depthBuffer[h, w])
            depth = far * near / (far - (far - near) * depthImg)

            depth /= math.cos(alpha)
            newTo = (depth / l) * vec + rf

            pixel2xy[w - center - dim // 2,
                     h - center - dim // 2] = newTo[:2]  #* had to flip
            if newTo[2] > 0.0:
                point_cloud = np.concatenate(
                    (point_cloud, newTo.reshape(1, 3)), axis=0)
    np.savez('pixel2xy20.npz', pixel2xy=pixel2xy)
    # print(pixel2xy)
    return pixel2xy


def getRayFromTo(mouseX, mouseY, params):

    width = params['imgW']
    height = params['imgH']
    # cameraUp = params['cameraUp']
    camForward = params['camForward']
    horizon = params['horizon']
    vertical = params['vertical']
    dist = params['dist']
    camTarget = params['camTarget']

    camPos = [
        camTarget[0] - dist * camForward[0],
        camTarget[1] - dist * camForward[1],
        camTarget[2] - dist * camForward[2]
    ]
    farPlane = 10000
    rayForward = [(camTarget[0] - camPos[0]), (camTarget[1] - camPos[1]),
                  (camTarget[2] - camPos[2])]
    lenFwd = math.sqrt(rayForward[0] * rayForward[0] +
                       rayForward[1] * rayForward[1] +
                       rayForward[2] * rayForward[2])
    invLen = farPlane * 1. / lenFwd
    rayForward = [
        invLen * rayForward[0], invLen * rayForward[1], invLen * rayForward[2]
    ]
    rayFrom = camPos
    oneOverWidth = float(1) / float(width)
    oneOverHeight = float(1) / float(height)

    dHor = [
        horizon[0] * oneOverWidth, horizon[1] * oneOverWidth,
        horizon[2] * oneOverWidth
    ]
    dVer = [
        vertical[0] * oneOverHeight, vertical[1] * oneOverHeight,
        vertical[2] * oneOverHeight
    ]
    # rayToCenter = [
    # rayFrom[0] + rayForward[0], rayFrom[1] + rayForward[1], rayFrom[2] + rayForward[2]
    # ]
    ortho = [
        -0.5 * horizon[0] + 0.5 * vertical[0] + float(mouseX) * dHor[0] -
        float(mouseY) * dVer[0], -0.5 * horizon[1] + 0.5 * vertical[1] +
        float(mouseX) * dHor[1] - float(mouseY) * dVer[1], -0.5 * horizon[2] +
        0.5 * vertical[2] + float(mouseX) * dHor[2] - float(mouseY) * dVer[2]
    ]

    rayTo = [
        rayFrom[0] + rayForward[0] + ortho[0],
        rayFrom[1] + rayForward[1] + ortho[1],
        rayFrom[2] + rayForward[2] + ortho[2]
    ]
    lenOrtho = math.sqrt(ortho[0] * ortho[0] + ortho[1] * ortho[1] +
                         ortho[2] * ortho[2])
    alpha = math.atan(lenOrtho / farPlane)
    return rayFrom, rayTo, alpha
