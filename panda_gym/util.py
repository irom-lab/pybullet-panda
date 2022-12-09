import numpy as np


def sample_uniform(rng, range, size=None):
    """Assume [low0, high0, low1, high1,...]"""
    num_range = int(len(range)/2)
    if num_range > 1:
        range_ind = rng.choice(num_range)
        range = range[range_ind*2:(range_ind+1)*2]
    return rng.uniform(range[0], range[1], size=size)


def sample_integers(rng, range, size=None):
    return rng.integers(range[0], range[1], size=size, endpoint=True)


def normalize_action(action, lower, upper):
    # assume action in [-1,1]
    return (action+1)/2*(upper-lower) + lower


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


# suppress pybullet print - from https://github.com/bulletphysics/bullet3/issues/2170
from ctypes import *
from contextlib import contextmanager
import sys, os
@contextmanager
def suppress_stdout():
    fd = sys.stdout.fileno()

    def _redirect_stdout(to):
        sys.stdout.close()  # + implicit flush()
        os.dup2(to.fileno(), fd)  # fd writes to 'to' file
        sys.stdout = os.fdopen(fd, "w")  # Python writes to fd

    with os.fdopen(os.dup(fd), "w") as old_stdout:
        with open(os.devnull, "w") as file:
            _redirect_stdout(to=file)
        try:
            yield  # allow code to be run with the redirected stdout
        finally:
            _redirect_stdout(to=old_stdout)  # restore stdout.
            # buffering and flags such as
            # CLOEXEC may be different
