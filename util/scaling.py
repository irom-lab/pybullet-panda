import numpy as np


def QuinticTimeScaling(Tf, t):
    """Computes s(t) for a quintic time scaling
	:param Tf: Total time of the motion in seconds from rest to rest
	:param t: The current time t satisfying 0 < t < Tf
	:return: The path parameter s(t) corresponding to a fifth-order
			 polynomial motion that begins and ends at zero velocity and zero
			 acceleration
	Example Input:
		Tf = 2
		t = 0.6
	Output:
		0.16308
	"""
    return 10 * (1.0 * t / Tf) ** 3 - 15 * (1.0 * t / Tf) ** 4 \
        + 6 * (1.0 * t / Tf) ** 5


def LinearTimeScaling(Tf, t):
    """
	Computes s(t) for a quintic time scaling
	"""
    return t / Tf


def traj_time_scaling(start_pos, end_pos, num_steps):
    traj_pos = np.zeros((num_steps, 3))
    for step in range(num_steps):
        s = 3 * (1.0 * step / num_steps)**2 - 2 * (1.0 * step / num_steps)**3
        traj_pos[step] = (end_pos-start_pos) * s + start_pos
    return traj_pos
