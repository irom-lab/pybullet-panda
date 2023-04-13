import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def rotate_tensor(orig_tensor, theta):
    """
	Rotate images clockwise.
	"""
    affine_mat = torch.tensor([[torch.cos(theta),
                                torch.sin(theta), 0],
                               [-torch.sin(theta),
                                torch.cos(theta), 0]]).view(2, 3, 1)
    affine_mat = affine_mat.permute(2, 0, 1).float().expand(
        orig_tensor.shape[0], -1, -1
    )  # repeat along batch size
    flow_grid = torch.nn.functional.affine_grid(
        affine_mat, orig_tensor.size(), align_corners=False
    ).to(orig_tensor.device)
    return torch.nn.functional.grid_sample(
        orig_tensor,
        flow_grid,
        mode='nearest',
        padding_mode='border',  # use border value to pad
        align_corners=False
    )


def save_affordance_map(img, pred, path_prefix):
    """
    Generate affordance map. Use RGB as background if available, otherwise stack depth to 3 channels.

    Args:
        img (Tensor): C,H,W, float32, [0,1]
        pred (Tensor): unnormalized
        path_prefix (str):
    """
    with torch.no_grad():
        C = img.shape[0]
        if C == 1:
            img_8bit = (img.detach().cpu().numpy() * 255).astype('uint8')[0]
            img_8bit = np.stack((img_8bit,) * 3, axis=-1)
            alpha = 0.5
        else:  # use RGB
            img_8bit = (img.detach().cpu().numpy()
                        * 255).astype('uint8')[1:].transpose(1, 2, 0)
            alpha = 0.2  # make heatmap less visible
        img_rgb = Image.fromarray(img_8bit, mode='RGB')
        # img_rgb.save(path_prefix + '_rgb.png')

        cmap = plt.get_cmap('jet')
        pred_detach = (torch.sigmoid(pred)).detach().cpu().numpy()
        pred_detach = (pred_detach - np.min(pred_detach)) / (
            np.max(pred_detach) - np.min(pred_detach)
        )  # normalize
        pred_cmap = cmap(pred_detach)
        pred_cmap = (np.delete(pred_cmap, 3, 2) * 255).astype('uint8')
        img_heat = Image.fromarray(pred_cmap, mode='RGB')
        # img_heat.save(path_prefix + '_heatmap.png')

        img_overlay = Image.blend(img_rgb, img_heat, alpha=alpha)
        img_overlay.save(path_prefix + '_overlay.png')


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
    rgb[:, :, 0] = r*a + (1.0-a) * R
    rgb[:, :, 1] = g*a + (1.0-a) * G
    rgb[:, :, 2] = b*a + (1.0-a) * B
    return np.asarray(rgb, dtype='uint8')
