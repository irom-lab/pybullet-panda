import numpy as np
import torch
from alano.train.vec_env import VecEnvBase


class VecEnvGraspMV(VecEnvBase):
    def __init__(self, venv, device, config_env):
        super(VecEnvGraspMV, self).__init__(venv, device, config_env)

        self.use_append = config_env.USE_APPEND

    def get_append(self, states):
        if self.use_append:
            method_args_list = [(state, ) for state in states]
            _append_all = self.venv.env_method_arg('_get_append',
                                                   method_args_list,
                                                   indices=range(self.n_envs))
            append_all = torch.FloatTensor(
                [append[0] for append in _append_all])
            return append_all.to(self.device)
        else:
            return None
        
class VecEnvGraspMVRandom(VecEnvGraspMV):
    def __init__(self, venv, device, config_env):
        super(VecEnvGraspMVRandom, self).__init__(venv, device, config_env)