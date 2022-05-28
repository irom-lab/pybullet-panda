
def sample_uniform(rng, range, size=None):
    """Assume [low0, high0, low1, high1,...]"""
    num_range = int(len(range)/2)
    if num_range > 1:
        range_ind = rng.choice(num_range)
        range = range[range_ind*2:(range_ind+1)*2]
    return rng.uniform(range[0], range[1], size=size)

def sample_integers(rng, range, size=None):
    return rng.integers(range[0], range[1], size=size, endpoint=True)
