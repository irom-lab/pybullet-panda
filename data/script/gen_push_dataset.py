import os
import numpy as np

from .util import sample_uniform
from panda_gym.push_env import PushEnv
from alano.geometry.transform import euler2quat
from alano.utils.pickle import save_obj



# Env for IK
env = PushEnv(task=None,
            renders=False,
            use_rgb=True,
            use_depth=True,
            #
            mu=0.3,
            sigma=0.01,
            camera_params=None)
env.reset()

#! seed train: 42, test: 424, ood: 4242
rng = np.random.default_rng(seed=4242)

save_tasks = []
num_task = 1000

obj_x_range = [0.54,0.54]   # slim_0/1 
obj_y_range = [0.0,0.0]
#* all half dims
obj_x_dim_range = [0.03, 0.03]  # slim_0/1
# obj_y_dim_range = [0.20, 0.20] # slim_1
obj_y_dim_range = [0.30, 0.30] # slim_2
obj_z_dim_range = [0.04, 0.04]
obj_yaw_range = [0, 0]
obj_density_range = [1e2, 1e2] # kg/m^3
obj_com_x_offset_range = [0, 0]

# obj_com_y_offset_range = [-0.18, -0.04, 0.04, 0.18] # slim_1
# obj_com_y_offset_range = [-0.20, -0.18, 0.18, 0.20] # slim_1_ood
# obj_com_y_offset_range = [-0.25, 0.25] # slim_2
obj_com_y_offset_range = [-0.30, -0.25, 0.25, 0.30] # slim_2_odd
ee_y_range = [-0.08, 0.08] # slim_1

# Generate
task_id = 0
while task_id < num_task:
    task = {}

    # Sample dimension
    obj_x_dim = sample_uniform(rng, obj_x_dim_range)
    obj_y_dim = sample_uniform(rng, obj_y_dim_range)
    obj_z_dim = sample_uniform(rng, obj_z_dim_range)

    # Sample pose
    obj_x = sample_uniform(rng, obj_x_range)
    obj_y = sample_uniform(rng, obj_y_range)
    obj_z = obj_z_dim + 0.005 # leave 1mm
    obj_pos = [obj_x, obj_y, obj_z]
    obj_yaw = sample_uniform(rng, obj_yaw_range)
    
    # Sample mass
    obj_density = sample_uniform(rng, obj_density_range)
    obj_mass = (obj_x_dim*2)*(obj_y_dim*2)*(obj_z_dim*2)*obj_density
    print(obj_mass)

    # Sample COM offset
    obj_com_x_offset = sample_uniform(rng, obj_com_x_offset_range)
    obj_com_y_offset = sample_uniform(rng, obj_com_y_offset_range)

    # Sample ee init
    ee_y = sample_uniform(rng, ee_y_range)

    # IK
    init_joint_angles = env.get_ik([0.40, ee_y, 0.18], euler2quat([0, 5*np.pi/6, 0]))[:7]
    print(init_joint_angles)

    # Add to task
    task['ee_y'] = ee_y
    task['obj_half_dim'] = [obj_x_dim, obj_y_dim, obj_z_dim]
    task['obj_pos'] = obj_pos
    task['obj_yaw'] = obj_yaw
    task['obj_mass'] = obj_mass
    task['obj_com_offset'] = [obj_com_x_offset, obj_com_y_offset, 0]
    task['init_joint_angles'] = init_joint_angles
    save_tasks += [task]
    task_id += 1

# Save
save_obj(save_tasks, os.path.join('data', 
                                  'private', 
                                  'box',
                                  'slim_' + str(num_task) + '_2_ood'))
