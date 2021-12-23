import os
from alano.utils.pickle import save_obj
import glob

urdf_parent_path = '/home/allen/data/processed_objects/YCB_simple/'
urdf_path_all = glob.glob(urdf_parent_path + '*.urdf')
save_tasks = []

for urdf_path in urdf_path_all:
    task = {}
    task['obj_path_list'] = [urdf_path]
    print(urdf_path)
    task['obj_init_state_all'] = [[0.5,0.0,0.0]]

# Save
save_obj(
    save_tasks,
    os.path.join('data', 'private', 'ycb_simple_60')
)
