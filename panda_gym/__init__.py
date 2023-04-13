from panda_gym.grasp_env import GraspEnv
from panda_gym.push_env import PushEnv
from panda_gym.push_tool_env import PushToolEnv
from panda_gym.lift_env import LiftEnv
from panda_gym.hammer_env import HammerEnv
from panda_gym.sweep_env import SweepEnv
from panda_gym.vec_env import VecEnvGrasp, VecEnvPush, VecEnvPushTool, VecEnvLift, VecEnvHammer, VecEnvSweep


env_dict = {
    'PushEnv': PushEnv,
    'PushToolEnv': PushToolEnv,
    'GraspEnv': GraspEnv,
    'LiftEnv': LiftEnv,
    'HammerEnv': HammerEnv,
    'SweepEnv': SweepEnv,
}

vec_env_dict = {
    'Push': VecEnvPush,
    'PushTool': VecEnvPushTool,
    'Grasp': VecEnvGrasp,
    'Lift': VecEnvLift,
    'Hammer': VecEnvHammer,
    'Sweep': VecEnvSweep,
}
