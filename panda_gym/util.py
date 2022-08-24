

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
