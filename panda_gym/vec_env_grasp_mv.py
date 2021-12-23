import numpy as np
import torch
from alano.train.vec_env import VecEnvBase

class VecEnvGraspMV(VecEnvBase):
    def __init__(self, venv, device, config_env):
        super(VecEnvGraspMV, self).__init__(venv, device, config_env)
