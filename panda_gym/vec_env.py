import torch
from stable_baselines3.common.vec_env import VecEnvWrapper

class VecEnvBase(VecEnvWrapper):
    def __init__(self, venv, device, config_env):
        super(VecEnvBase, self).__init__(venv)
        self.device = device
        self.n_envs = self.num_envs
        self.config_env = config_env

    def reset(self, tasks):
        args_all = [(task, ) for task in tasks]
        obs = self.venv.reset_arg(args_all)
        return torch.from_numpy(obs).to(self.device)

    def reset_one(self, index, task):
        obs = self.venv.env_method('reset', task=task, indices=[index])[0]
        return torch.from_numpy(obs).to(self.device)

    def get_obs(self, states):
        method_args_list = [(state, ) for state in states]
        observations = torch.FloatTensor(
            self.venv.env_method_arg('_get_obs',
                                     method_args_list=method_args_list,
                                     indices=range(self.n_envs)))
        return observations.to(self.device)

    def step_async(self, actions):
        if isinstance(actions, torch.LongTensor):
            # Squeeze the dimension for discrete actions
            actions = actions.squeeze(1)
        actions = actions.cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        obs = torch.from_numpy(obs).to(self.device)
        reward = torch.from_numpy(reward).unsqueeze(dim=1).float()
        return obs, reward, done, info


class VecEnvPanda(VecEnvBase):
    def __init__(self, venv, device, config_env):
        super(VecEnvPanda, self).__init__(venv, device, config_env)

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

class VecEnvGraspMV(VecEnvPanda):
    def __init__(self, venv, device, config_env):
        super(VecEnvGraspMV, self).__init__(venv, device, config_env)

class VecEnvGraspMVRandom(VecEnvPanda):
    def __init__(self, venv, device, config_env):
        super(VecEnvGraspMVRandom, self).__init__(venv, device, config_env)
        
class VecEnvPush(VecEnvPanda):
    def __init__(self, venv, device, config_env):
        super(VecEnvPush, self).__init__(venv, device, config_env)
