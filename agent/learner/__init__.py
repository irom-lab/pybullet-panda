from .grasp_bandit import GraspBandit
from .grasp_bandit_eq import GraspBanditEq
from .grasp_script import GraspScript
from .grasp_script_flip import GraspScriptFlip


def get_learner(name):
    if name == 'GraspBandit':
        return GraspBandit
    elif name == 'GraspBanditEq':
        return GraspBanditEq
    elif name == 'GraspScript':
        return GraspScript
    elif name == 'GraspScriptFlip':
        return GraspScriptFlip
    else:
        raise 'Unknown learner type!'
