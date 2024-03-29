import numpy as np
import logging
import torch
import random

from util.numeric import normalize


class UtilGraspFlip():
    """
    Utilities for the flipped grasping environment.
    """
    def __init__(self, cfg):
        self.use_extra = cfg.use_extra
        random.seed(cfg.seed)


    def get_extra(self, tasks):
        table_rgba_all = []
        for task in tasks:
            # scaling_all += [task['global_scaling']]
            table_rgba_all += [task['table_rgba']]
        return np.array(table_rgba_all)
