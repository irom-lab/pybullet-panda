import torch
from alano.train.vec_env import VecEnvBase


class VecEnvPanda(VecEnvBase):
    def __init__(self, venv, cpu_offset, device, config_env):
        super(VecEnvPanda, self).__init__(venv, cpu_offset, device, config_env)

        self.use_append = config_env.USE_APPEND

    # def get_append(self, states):
    #     if self.use_append:
    #         method_args_list = [(state, ) for state in states]
    #         _append_all = self.venv.env_method_arg('_get_append',
    #                                                method_args_list,
    #                                                indices=range(self.n_envs))
    #         append_all = torch.FloatTensor(
    #             [append[0] for append in _append_all])
    #         return append_all.to(self.device)
    #     else:
    #         return None

    @property
    def state_dim(self):
        return self.get_attr('state_dim', indices=[0])[0]


    @property
    def action_dim(self):
        return self.get_attr('action_dim', indices=[0])[0]


class VecEnvGraspMV(VecEnvPanda):
    def __init__(self, venv, cpu_offset, device, config_env):
        super(VecEnvGraspMV, self).__init__(venv, cpu_offset, device, config_env)


class VecEnvGraspMVRandom(VecEnvPanda):
    def __init__(self, venv, cpu_offset, device, config_env):
        super(VecEnvGraspMVRandom, self).__init__(venv, cpu_offset, device, config_env)


class VecEnvPush(VecEnvPanda):
    def __init__(self, venv, cpu_offset, device, config_env):
        super(VecEnvPush, self).__init__(venv, cpu_offset, device, config_env)


class VecEnvPushTool(VecEnvPanda):
    def __init__(self, venv, cpu_offset, device, config_env):
        super(VecEnvPushTool, self).__init__(venv, cpu_offset, device, config_env)


class VecEnvLift(VecEnvPanda):
    def __init__(self, venv, cpu_offset, device, config_env):
        super(VecEnvLift, self).__init__(venv, cpu_offset, device, config_env)


class VecEnvHammer(VecEnvPanda):
    def __init__(self, venv, cpu_offset, device, config_env):
        super(VecEnvHammer, self).__init__(venv, cpu_offset, device, config_env)


class VecEnvSweep(VecEnvPanda):
    def __init__(self, venv, cpu_offset, device, config_env):
        super(VecEnvSweep, self).__init__(venv, cpu_offset, device, config_env)



