from panda_gym.push_env import PushEnv
from panda_gym.grasp_env import GraspEnv
from panda_gym.grasp_flip_env import GraspFlipEnv
from panda_gym.vec_env import VecEnvPush, VecEnvGrasp

from omegaconf import OmegaConf


def get_env(name):
    if name == 'Push-v0':
        return PushEnv
    elif name == 'Grasp-v0':
        return GraspEnv
    elif name == 'GraspFlip-v0':
        return GraspFlipEnv
    else:
        raise 'Unknown env type!'


def get_vec_env(name):
    if name == 'Push-v0':
        return VecEnvPush
    elif name == 'Grasp-v0' or name == 'GraspFlip-v0':
        return VecEnvGrasp
    else:
        raise 'Unknown vec env type!'


def get_vec_env_cfg(name, cfg_env):
    vec_env_cfg = cfg_env.specific
    return vec_env_cfg

