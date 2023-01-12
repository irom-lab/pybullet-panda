from .grasp_bandit import GraspBandit
from .grasp_bandit_eq import GraspBanditEq
from .grasp_script import GraspScript


def get_learner(name):
    if name == 'GraspBandit':
        return GraspBandit
    elif name == 'GraspScript':
        return GraspScript
    elif name == 'GraspBanditEq':
        return GraspBanditEq
    else:
        raise 'Unknown learner type!'
