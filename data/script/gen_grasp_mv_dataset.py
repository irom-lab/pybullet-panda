import os
import glob
import random
import numpy as np
import itertools
import math
from util.misc import save_obj, load_obj

random.seed(1000)
np.random.seed(1000)

# urdf_parent_path = '/home/allen/data/processed_objects/YCB_simple/'
# urdf_parent_path = '/home/allen/data/wasserstein/grasp/primitive_box_v2/'
# urdf_parent_path = '/home/allen/data/processed_objects/SNC_v4_mug/' # 1078
urdf_parent_path = '/home/allen/data/processed_objects/SNC_v4/03797390/' # 49
obj_id_list = [10,11,12,13,19,28,31,33,34,51,62,67,68,71,72,73,78,83,84,88,1,3,6,7,9,15,16,20,23,41,47,50,52,54,56,57,61,65,74,77,80,82,86,92,89,91,94,97,98]

urdf_path_all = glob.glob(urdf_parent_path + '*.urdf')
save_tasks = []
num_task = 5000
num_obj_per_task = 3    #!
# obj_x_range = [0.44,0.56] # upright
# obj_y_range = [-0.06,0.06]
obj_x_range = [0.42,0.58]   # upright_new
obj_y_range = [-0.08,0.08]
# obj_x_range = [0.45,0.55] # random
# obj_y_range = [-0.05,0.05]
obj_z_range = [0.15,0.15]
obj_yaw_range = [-np.pi, np.pi]
obj_pitch_range = [0, 0]
obj_roll_range = [0, 0]
# obj_pitch_range = [-np.pi/3, -np.pi/6]   # random
# obj_roll_range = [-np.pi/3, -np.pi/6]
obj_scale_range = [0.8, 1.0]

# Load height and radius
obj_height_all = load_obj(os.path.join(urdf_parent_path, 'obj_height_all'))
obj_radius_all = load_obj(os.path.join(urdf_parent_path, 'obj_radius_all'))

# Generate
max_num_attempt = 100000
task_id = 0
while task_id < num_task:
    task = {}
    
    # Sample object
    obj_id_task = random.sample(obj_id_list, num_obj_per_task)    # without replacement
    task['obj_path_list'] = [urdf_parent_path + str(obj_id) + '.urdf' for obj_id in obj_id_task]

    # Sample scale
    obj_scale_task = np.random.uniform(low=obj_scale_range[0], high=obj_scale_range[1], size=(num_obj_per_task,))
    task['obj_scale_all'] = obj_scale_task

    # Save height
    task['obj_height_all'] = [obj_height_all[obj_id] for obj_id in obj_id_task]

    # Get radius
    obj_radius_task = [obj_scale_task[ind] * obj_radius_all[obj_id] for ind, obj_id in enumerate(obj_id_task)]

    # Sample pose
    comb_task = list(itertools.combinations(range(num_obj_per_task), 2))
    obj_target_dist_task = [obj_radius_task[ind_1] + obj_radius_task[ind_2] for ind_1, ind_2 in comb_task]   # add a bit room
    num_attempt = 0
    while num_attempt < max_num_attempt:
        num_attempt += 1
        obj_x_task = np.random.uniform(low=obj_x_range[0], high=obj_x_range[1], size=(num_obj_per_task, 1)) 
        obj_y_task = np.random.uniform(low=obj_y_range[0], high=obj_y_range[1], size=(num_obj_per_task, 1))
        obj_z_task = np.random.uniform(low=obj_z_range[0], high=obj_z_range[1], size=(num_obj_per_task, 1))
        obj_yaw_task = np.random.uniform(low=obj_yaw_range[0], high=obj_yaw_range[1], size=(num_obj_per_task, 1))
        obj_pitch_task = np.random.uniform(low=obj_pitch_range[0], high=obj_pitch_range[1], size=(num_obj_per_task, 1))
        obj_roll_task = np.random.uniform(low=obj_roll_range[0], high=obj_roll_range[1], size=(num_obj_per_task, 1)) 
        obj_pose_all = np.concatenate((obj_x_task, obj_y_task, obj_z_task, obj_yaw_task, obj_pitch_task, obj_roll_task), axis=1)
        
        flag = True
        for comb_ind, (ind_1, ind_2) in enumerate(comb_task):
            obj_loc_1 = [obj_x_task[ind_1], obj_y_task[ind_1]]
            obj_loc_2 = [obj_x_task[ind_2], obj_y_task[ind_2]]
            obj_dist = math.sqrt((obj_loc_2[0] - obj_loc_1[0])**2 + (obj_loc_2[1] - obj_loc_1[1])**2)
            # if obj_dist < 0.05:
            if obj_dist < obj_target_dist_task[comb_ind]:
                flag = False
                break
        if not flag:
            continue
        else:
            break
    if not flag:    # do not save
        continue
    print('here', num_attempt, task_id)
        
    task['obj_state_all'] = obj_pose_all  # xyz, yaw, pitch, roll
    
    # Sample color
    obj_rgb_all = np.random.uniform(low=0, high=1, size=(num_obj_per_task, 3))
    obj_rgba_all = np.hstack((obj_rgb_all, np.ones((num_obj_per_task, 1))))
    task['obj_rgba_all'] = obj_rgba_all

    # Add to task
    save_tasks += [task]
    task_id += 1

# for urdf_path in urdf_path_all:
#     task = {}
#     task['obj_path_list'] = [urdf_path]
#     print(urdf_path)
#     task['obj_init_state_all'] = [[0.5, 0.0, 0.0, 0.0]]  # xyz, yaw
#     save_tasks += [task]

# Save
save_obj(save_tasks, os.path.join('data', 'private', 'mug','mug_3_5000_upright_prior'))
