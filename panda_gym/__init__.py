# import gym

# gym.envs.register(  # no time limit imposed
#     id='GraspMultiView-v0',
#     entry_point='panda_gym.grasp_mv_env:GraspMultiViewEnv',
# )

# gym.envs.register(  # no time limit imposed
#     id='GraspMultiViewRandom-v0',
#     entry_point='panda_gym.grasp_mv_random_env:GraspMultiViewRandomEnv',
# )

# gym.envs.register(  # no time limit imposed
#     id='Push-v0',
#     entry_point='panda_gym.push_env:PushEnv',
# )

# gym.envs.register(  # no time limit imposed
#     id='PushTool-v0',
#     entry_point='panda_gym.push_tool_env:PushToolEnv',
# )

# gym.envs.register(  # no time limit imposed
#     id='Lift-v0',
#     entry_point='panda_gym.lift_env:LiftEnv',
# )

# gym.envs.register(  # no time limit imposed
#     id='Hammer-v0',
#     entry_point='panda_gym.hammer_env:HammerEnv',
# )

# gym.envs.register(  # no time limit imposed
#     id='Sweep-v0',
#     entry_point='panda_gym.sweep_env:SweepEnv',
# )

from .push_env import PushEnv
from .grasp_env import GraspEnv
from .vec_env import VecEnvPush, VecEnvGrasp

from omegaconf import OmegaConf


def get_env(name):
    if name == 'Push-v0':
        return PushEnv
    elif name == 'Grasp-v0':
        return GraspEnv
    else:
        raise 'Unknown env type!'


def get_vec_env(name):
    if name == 'Push-v0':
        return VecEnvPush
    elif name == 'Grasp-v0':
        return VecEnvGrasp
    else:
        raise 'Unknown vec env type!'


def get_vec_env_cfg(name, cfg_env):
    vec_env_cfg = cfg_env.specific
    return vec_env_cfg

