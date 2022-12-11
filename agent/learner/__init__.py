from .grasp_bandit import GraspBandit


def get_learner(name):
    if name == 'GraspBandit':
        return GraspBandit
    else:
        raise 'Unknown learner type!'
