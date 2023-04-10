import os
import numpy as np
import random
from omegaconf import OmegaConf

from util.numeric import sample_uniform
from util.geom import euler2quat
from util.misc import save_obj

seed = 42
random.seed(42)
rng = np.random.default_rng(seed=seed)

save_tasks = []
num_task = 500
# obj_x_range = [0.40, 0.60]
# obj_y_range = [-0.1, 0.1]
obj_x_range = [0.45, 0.45]
obj_y_range = [-0.1, -0.1]
obj_yaw_range = [np.pi/4, np.pi/4]  # rotate gavel (yaw=0 means handle pointing in positive y) so a grasp of theta=0 can grasp the handle
obj_all = {'mug': ['data/sample/mug/3.urdf', 0.15],
           'gavel': ['data/sample/42_gavel/42_gavel.urdf', 0.05],}  # obj_path, default_z_pos
table_rgba_choices = [[0.3,0.3,0.3,1], [0.7,0.7,0.7,1]]

# Generate
task_id = 0
while task_id < num_task:
    task = OmegaConf.create()

    # Sample object
    obj_name = random.sample(obj_all.keys(), 1)[0]
    
    # Make mug bigger
    if obj_name == 'mug':
        global_scaling = 1.2
        y_pos = -0.1
    else:
        global_scaling = 1.0
        y_pos = 0.1
    z_pos = obj_all[obj_name][1] * global_scaling

    # Sample pose
    # x_pos = random.sample([0.45,0.55], 1)[0]
    x_pos = sample_uniform(rng, obj_x_range)
    # y_pos = sample_uniform(rng, obj_y_range)
    yaw = sample_uniform(rng, obj_yaw_range)
    
    # Sample table color
    table_rgba = random.sample(table_rgba_choices, 1)[0]

    # Add to task
    task_obj_name = obj_name
    task.obj_pos = [x_pos, y_pos, z_pos]
    task.obj_path = obj_all[obj_name][0]
    task.obj_quat = euler2quat([yaw, 0, 0]).tolist() # (yaw, pitch, roll), instrinsic
    task.global_scaling = global_scaling
    task.table_rgba = table_rgba
    save_tasks += [task]
    task_id += 1

# Save
save_obj(save_tasks, os.path.join('data', 'mug', 'eq_3'))
