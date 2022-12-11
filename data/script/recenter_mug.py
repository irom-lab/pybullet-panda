import trimesh
import random
from shutil import copyfile
import numpy as np
from alano.geometry.urdf import save_simple_urdf
from alano.geometry.mesh import watertight_simplify
import pybullet as p
import pickle
import os
from alano.util.pickle import save_obj


obj_id_list = [10,11,12,13,19,28,31,33,34,51,62,67,68,71,72,73,78,83,84,88,1,3,6,7,9,15,16,20,23,41,47,50,52,54,56,57,61,65,74,77,80,82,86,92,89,91,94,97,98]
load_folder_path = '/home/allen/data/processed_objects/SNC_v4/03797390/'
num_obj_orig = len(obj_id_list)
random.shuffle(obj_id_list)
print('Number of original objects:', num_obj_orig)

# ===================== Get dimensions ===================#
obj_radius_all = {}
obj_height_all = {}
for obj_id in obj_id_list:
    obj_path_orig = load_folder_path + str(obj_id) + '_centered.obj'
    mesh = trimesh.load(obj_path_orig)

    # Save
    obj_height_all[obj_id] = mesh.bounds[1,2] - mesh.bounds[0,2]
    obj_radius_all[obj_id] = max(mesh.bounds[1,0] - mesh.bounds[0,0], mesh.bounds[1,1] - mesh.bounds[0,1]) / 2
print(obj_height_all, obj_radius_all)
save_obj(obj_height_all, os.path.join(load_folder_path, 'obj_height_all'))
save_obj(obj_radius_all, os.path.join(load_folder_path, 'obj_radius_all'))
while 1:
    continue

# ===================== Process raw object ===================#
obj_height_all = {}
for obj_id in obj_id_list:
    obj_path_orig = load_folder_path + str(obj_id) + '_centered.obj'
    mesh = trimesh.load(obj_path_orig)

    # Watertight, simplify
    obj_path_wts = watertight_simplify(path=load_folder_path, name=str(obj_id))
    obj_path_wts = load_folder_path + obj_path_wts + '.obj'

    # Use actual center of mass as origin
    mesh_wts = trimesh.load(obj_path_wts)
    matrix = np.array([[1,0,0,-mesh_wts.center_mass[0]], 
                       [0,1,0,-mesh_wts.center_mass[1]],
                       [0,0,1,-mesh_wts.center_mass[2]],
                       [0,0,0,1]])
    mesh_wts.apply_transform(matrix)
    obj_height_all[str(id)] = mesh_wts.center_mass[-1]

    # Save
    obj_path_centered = load_folder_path + str(obj_id) + '_centered.obj'
    mesh_wts.export(obj_path_centered)
    
    # Decompose using trimesh and urdf
    os.makedirs(load_folder_path + str(obj_id) + '/')
    trimesh.exchange.urdf.export_urdf(mesh, 
                                    load_folder_path, 
                                    ind=obj_id,
                                    mass=0.3,
                                    keep_concave_part=False)

    # decomposition
    obj_path_final = load_folder_path + str(obj_id) + '_final.obj'
    p.vhacd(obj_path_centered, obj_path_final, fileNameLogging='data/private/'+str(obj_id)+'_log.txt',resolution=16000000, minVolumePerCH=0.0000000001, maxNumVerticesPerCH=1024)
print('done ')
print(obj_height_all)
with open(load_folder_path + 'obj_height_all.pkl', 'wb') as f:
    pickle.dump(obj_height_all, f, pickle.HIGHEST_PROTOCOL)
while 1:
    continue

# ===================== Save urdf ===================#
save_folder_name = '/home/allen/data/processed_objects/SNC_v4_mug/'
save_obj_ind = 0
save_urdf_ind = 0

# Randomize urdf index
num_scale_per_obj = 22
num_urdf = num_obj_orig*num_scale_per_obj
urdf_index_all = list(range(num_urdf))
random.shuffle(urdf_index_all)

# Load height all
# with open(load_folder_path + 'obj_height_all.pkl', 'rb') as f:
#     obj_height_all = pickle.load(f)
# urdf_height_all = [0]*num_urdf

# Loop through each object category
for obj_id in obj_id_list:
    print('Processing', obj_id)	

    # Set up paths for loading and saving objects
    load_obj = load_folder_path + str(obj_id) + '_final.obj'
    save_obj = save_folder_name + str(save_obj_ind) + '.obj'
    # obj_height = obj_height_all[str(obj_id)]

    # Copy decom_wt obj to new directory with new index
    copyfile(load_obj, save_obj)

    # Scale into multiple mugs
    for scale_ind in range(num_scale_per_obj):
        xy_scale = np.random.uniform(0.80, 1.10, (1,))  # was 0.9 and 1.2 for SNC_v4_mug, 0.9 and 1.1 for SNC_v4_mug_small
        z_scale = np.random.uniform(0.80, 1.20, (1,))

        # Get ind for urdf
        urdf_ind = urdf_index_all[save_urdf_ind]

        # Save URDF for decom
        save_simple_urdf(path=save_folder_name, 
                urdf_name=str(urdf_ind), 
                mesh_name=str(save_obj_ind)+'.obj', 
                x_scale=xy_scale, 
                y_scale=xy_scale, 
                z_scale=z_scale)

        # Save urdf height
        # urdf_height_all[urdf_ind] = obj_height 

        # Increment urdf index
        save_urdf_ind += 1

    # Increment new index
    save_obj_ind += 1

# Save urdf height
# with open(save_folder_name + 'urdf_height_all.pkl', 'wb') as f:
#     pickle.dump(urdf_height_all, f, pickle.HIGHEST_PROTOCOL)
print('Number of processed:', save_urdf_ind)
