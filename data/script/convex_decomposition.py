import time
import pybullet as p

load_name = 'franka/meshes/collision/franka_finger_lower_wide_flat.obj'
save_decom_name = 'franka/meshes/collision/franka_finger_lower_wide_flat_decom.obj'

start_time = time.time()
p.vhacd(load_name,
        save_decom_name,
        fileNameLogging='log/' + str(0) + '_log.txt',
        resolution=16000000,
        minVolumePerCH=0.0000000001,
        maxNumVerticesPerCH=1024)
print('Time took to decompose: ', time.time() - start_time)
