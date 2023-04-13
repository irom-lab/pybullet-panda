import math


def sample_uniform(rng, range, size=None):
    """Assume [low0, high0, low1, high1,...]"""
    num_range = int(len(range) / 2)
    if num_range > 1:
        range_ind = rng.choice(num_range)
        range = range[range_ind * 2:(range_ind+1) * 2]
    return rng.uniform(range[0], range[1], size=size)


def sample_integers(rng, range, size=None):
    return rng.integers(range[0], range[1], size=size, endpoint=True)


def unnormalize_tanh(data, lb, ub):
    return (data/2 + 0.5) * (ub-lb) + lb


def normalize(data, lb, ub):
    """To [0,1]"""
    return (data-lb) / (ub-lb)


def unnormalize(data, lb, ub):
    """From [0,1]"""
    return (ub-lb) * data + lb


def wrap_angle(value, low, high):
    assert low < high
    width = high - low
    return value - width * math.floor((value-low) / width)
