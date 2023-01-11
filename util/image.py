import torch
import numpy as np


def rotate_tensor(orig_tensor, theta):
    """
	Rotate images clockwise
	"""
    affine_mat = torch.tensor([[torch.cos(theta), torch.sin(theta), 0],
                               [-torch.sin(theta), torch.cos(theta), 0]]).view(2,3,1)
    affine_mat = affine_mat.permute(2, 0, 1).float().expand(orig_tensor.shape[0], -1, -1)   # repeat along batch size
    flow_grid = torch.nn.functional.affine_grid(affine_mat,
                                                orig_tensor.size(),
                                                align_corners=False).to(orig_tensor.device)
    return torch.nn.functional.grid_sample(orig_tensor,
                                           flow_grid,
                                           mode='nearest',
                                           align_corners=False)


def rgba2rgb(rgba, background=(255, 255, 255)):
    """
    Convert rgba to rgb.

    Args:
        rgba (tuple):
        background (tuple):

    Returns:
        rgb (tuple):
    """
    row, col, ch = rgba.shape
    if ch == 3:
        return rgba
    assert ch == 4, 'RGBA image has 4 channels.'

    rgb = np.zeros((row, col, 3), dtype='float32')
    r, g, b, a = rgba[:, :, 0], rgba[:, :, 1], rgba[:, :, 2], rgba[:, :, 3]
    a = np.asarray(a, dtype='float32') / 255.0
    R, G, B = background
    rgb[:, :, 0] = r * a + (1.0 - a) * R
    rgb[:, :, 1] = g * a + (1.0 - a) * G
    rgb[:, :, 2] = b * a + (1.0 - a) * B
    return np.asarray(rgb, dtype='uint8')
