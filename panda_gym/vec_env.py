from panda_gym.util.vec_env import VecEnvBase


class VecEnvPanda(VecEnvBase):

    def __init__(self, venv, cpu_offset, device, cfg):
        super(VecEnvPanda, self).__init__(venv, cpu_offset, device)
        self.cfg = cfg

    @property
    def state_dim(self):
        return self.get_attr('state_dim', indices=[0])[0]

    @property
    def action_dim(self):
        return self.get_attr('action_dim', indices=[0])[0]


class VecEnvGrasp(VecEnvPanda):

    def __init__(self, venv, cpu_offset, device, cfg):
        super(VecEnvGrasp, self).__init__(venv, cpu_offset, device, cfg)


class VecEnvGraspMV(VecEnvPanda):

    def __init__(self, venv, cpu_offset, device, cfg):
        super(VecEnvGraspMV, self).__init__(venv, cpu_offset, device, cfg)


class VecEnvGraspMVRandom(VecEnvPanda):

    def __init__(self, venv, cpu_offset, device, cfg):
        super(VecEnvGraspMVRandom,
              self).__init__(venv, cpu_offset, device, cfg)


class VecEnvPush(VecEnvPanda):

    def __init__(self, venv, cpu_offset, device, cfg):
        super(VecEnvPush, self).__init__(venv, cpu_offset, device, cfg)


class VecEnvPushTool(VecEnvPanda):

    def __init__(self, venv, cpu_offset, device, cfg):
        super(VecEnvPushTool, self).__init__(venv, cpu_offset, device, cfg)


class VecEnvLift(VecEnvPanda):

    def __init__(self, venv, cpu_offset, device, cfg):
        super(VecEnvLift, self).__init__(venv, cpu_offset, device, cfg)


class VecEnvHammer(VecEnvPanda):

    def __init__(self, venv, cpu_offset, device, cfg):
        super(VecEnvHammer, self).__init__(venv, cpu_offset, device, cfg)


class VecEnvSweep(VecEnvPanda):

    def __init__(self, venv, cpu_offset, device, cfg):
        super(VecEnvSweep, self).__init__(venv, cpu_offset, device, cfg)
