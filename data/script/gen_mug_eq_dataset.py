import os
import numpy as np
import random

from util.numeric import sample_uniform
from util.geom import euler2quat
from util.misc import save_obj

seed = 42
random.seed(42)
rng = np.random.default_rng(seed=seed)

save_tasks = []
num_task = 500

# obj_x_range = [0.54,0.54]   # slim_0/1 
obj_y_range = [-0.1, 0.1]
obj_yaw_range = [0, 0]  # no yaw right now
default_z_pos = 0.15

# Generate
task_id = 0
while task_id < num_task:
    task = {}
    
    if task_id % 2 == 0:    # left, small
        global_scaling = 1
        x_pos = 0.55
    else:                   # right, large
        global_scaling = 1.2
        x_pos = 0.45
    z_pos = default_z_pos * global_scaling

    # Sample dimension
    x_pos = random.sample([0.45,0.55], 1)[0]
    y_pos = sample_uniform(rng, obj_y_range)

    # Add to task
    task['obj_pos'] = [x_pos, y_pos, z_pos]
    task['obj_path'] = 'data/sample/mug/3.urdf'
    task['obj_quat'] = [0,0,0,1]
    task['global_scaling'] = global_scaling
    save_tasks += [task]
    task_id += 1

# Save
save_obj(save_tasks, os.path.join('data', 'mug', 'eq_0'))
