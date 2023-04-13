import numpy as np
import numpy as np
import pybullet as p
from util.geom import quat2rot


def full_jacob_pb(jac_t, jac_r):
    return np.vstack(
        (jac_t[0], jac_t[1], jac_t[2], jac_r[0], jac_r[1], jac_r[2])
    )


def plot_frame_pb(pos, orn=np.array([0., 0., 0., 1.]), w_first=False):
    rot = quat2rot(orn, w_first)
    endPos = pos + 0.1 * rot[:, 0]
    p.addUserDebugLine(pos, endPos, lineColorRGB=[1, 0, 0], lineWidth=5)
    endPos = pos + 0.1 * rot[:, 1]
    p.addUserDebugLine(pos, endPos, lineColorRGB=[0, 1, 0], lineWidth=5)
    endPos = pos + 0.1 * rot[:, 2]
    p.addUserDebugLine(pos, endPos, lineColorRGB=[0, 0, 1], lineWidth=5)


def plot_line_pb(p1, p2, lineColorRGB=[1, 0, 0], lineWidth=5):
    p.addUserDebugLine(p1, p2, lineColorRGB=lineColorRGB, lineWidth=lineWidth)
