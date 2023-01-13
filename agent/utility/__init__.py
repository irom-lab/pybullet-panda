from .util_grasp_flip import UtilGraspFlip


def get_utility(name):
    if name == 'GraspFlip':
        return UtilGraspFlip
    elif name == 'Dummy':
        return UtilDummy
    else:
        raise 'Unknown utility type!'


class UtilDummy():
    """
    Dummy utility class
    """
    def __init__(self, cfg):
        self.use_extra = False
