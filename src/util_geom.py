from src import *


def euler2quat(a):

    yaw = a[1]
    pitch = a[0]  # flipped
    roll = a[2]

    c1 = np.cos(yaw / 2)
    s1 = np.sin(yaw / 2)
    c2 = np.cos(pitch / 2)
    s2 = np.sin(pitch / 2)
    c3 = np.cos(roll / 2)
    s3 = np.sin(roll / 2)
    c1c2 = c1 * c2
    s1s2 = s1 * s2
    w = c1c2 * c3 - s1s2 * s3
    x = c1c2 * s3 + s1s2 * c3
    y = s1 * c2 * c3 + c1 * s2 * s3
    z = c1 * s2 * c3 - s1 * c2 * s3

    return np.array([x, y, z, w])
    # return rot2quat(euler2rot(a))


def quatMult(p, q):
    """
	Multiply two quaternions (a,b,c,w)
	"""

    w = p[3] * q[3] - np.dot(p[:3], q[:3])
    abc = p[3] * q[:3] + q[3] * p[:3] + np.cross(p[:3], q[:3])
    return np.hstack((abc, w))


def rotate_tensor(orig_tensor, theta):
    """
	Rotate images clockwise
	"""
    affine_mat = np.array([[np.cos(theta), np.sin(theta), 0],
                           [-np.sin(theta), np.cos(theta), 0]])
    affine_mat.shape = (2, 3, 1)
    affine_mat = torch.from_numpy(affine_mat).permute(2, 0, 1).float()
    flow_grid = torch.nn.functional.affine_grid(affine_mat,
                                                orig_tensor.size(),
                                                align_corners=False)
    return torch.nn.functional.grid_sample(orig_tensor,
                                           flow_grid,
                                           mode='nearest',
                                           align_corners=False)
