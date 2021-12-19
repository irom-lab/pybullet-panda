import os
import random
import math
import numpy as np
from numpy import array
import matplotlib.pyplot as plt
import torch
import pybullet as p
from PIL import Image
import concurrent.futures
import psutil

from src.fcn import MLP
from src.util_depth import getParameters
from src.util_misc import save__init__args, ensure_directory
from panda.util_geom import quatMult, euler2quat
from panda.panda_env import PandaEnv


class GraspSim:
    def __init__(self, img_size=96, hidden_size=200, max_obj_height=0.05):
        save__init__args(locals())

        # Height of the EE before and after reaching down
        self.min_ee_z = 0.15  # EE height when fingers contact the table

        # Height of grasp from the depth at the chosen pixel
        self.delta_z = 0.03

        # Initialize panda env
        self.mu = 0.3
        self.sigma = 0.03
        self.panda_env = PandaEnv(mu=self.mu,
                                  sigma=self.sigma,
                                  finger_type='long')
        self.obj_id_list = []
        self.max_obj_height = max_obj_height

        # Pixel to xy
        pixel_xy_path = 'data/pixel2xy' + str(img_size) + '.npz'
        self.pixel2xy_mat = np.load(pixel_xy_path)['pixel2xy']  # HxWx2

        # Initialize model
        self.policy = MLP(hidden_size=hidden_size,
                          img_size=self.img_size).to('cpu')
        self.policy.eval()

    def update_policy(self, model_dict):
        self.policy.load_state_dict(model_dict)

    def load_obj(self, obj_path_list, obj_height_list):
        self.obj_id_list = []  # reinitialize
        self.obj_initial_height_list = {}
        env_x = [0.48, 0.53]  #!
        env_y = [-0.03, 0.02]
        env_yaw = [-3.14, 3.14]
        num_obj = len(obj_path_list)

        obj_x_initial = np.random.uniform(low=env_x[0],
                                          high=env_x[1],
                                          size=(num_obj, ))
        obj_y_initial = np.random.uniform(low=env_y[0],
                                          high=env_y[1],
                                          size=(num_obj, ))
        obj_orn_initial_all = np.random.uniform(low=env_yaw[0],
                                                high=env_yaw[1],
                                                size=(num_obj, 3))
        obj_orn_initial_all[:, :-1] = 0

        for obj_ind in range(num_obj):
            pos = [
                obj_x_initial[obj_ind], obj_y_initial[obj_ind],
                obj_height_list[obj_ind] / 2 + 0.001
            ]
            obj_id = p.loadURDF(obj_path_list[obj_ind],
                                basePosition=pos,
                                baseOrientation=p.getQuaternionFromEuler(
                                    obj_orn_initial_all[obj_ind]))
            self.obj_id_list += [obj_id]

            # Infer number of links - change dynamics for each
            num_joint = p.getNumJoints(obj_id)
            link_all = [-1] + [*range(num_joint)]
            for link_id in link_all:
                p.changeDynamics(
                    obj_id,
                    link_id,
                    lateralFriction=self.mu,
                    spinningFriction=self.sigma,
                    frictionAnchor=1,
                )

        # Let objects settle (actually do not need since we know the height of object and can make sure it spawns very close to table level)
        for _ in range(10):
            p.stepSimulation()

        # Record object initial height (for comparing with final height when checking if lifted). Note that obj_initial_height_list is a dict
        for obj_id in self.obj_id_list:
            pos, _ = p.getBasePositionAndOrientation(obj_id)
            self.obj_initial_height_list[obj_id] = pos[2]

    def sim_parallel(self,
                     obj_path_list,
                     obj_height_list,
                     eps=0,
                     num_cpus=16,
                     cpu_offset=0):
        num_trial = len(obj_path_list)

        # Determine how many trials will be epsilon-random
        random_chosen_ids = random.sample(range(len(obj_path_list)),
                                          k=int(len(obj_path_list) *
                                                eps))  # round down
        random_all = np.zeros((len(obj_path_list)))
        random_all[random_chosen_ids] = 1

        # Make each path as a list
        obj_path_list = [[obj_path] for obj_path in obj_path_list]
        obj_height_list = [[obj_height] for obj_height in obj_height_list]

        # Split for each worker
        trial_ind_batch_all = np.array_split(np.arange(num_trial), num_cpus)

        # Construct args - one cpu per worker
        args = (
            ([obj_path_list[id] for id in trial_ind_batch
              ], [obj_height_list[id] for id in trial_ind_batch],
             [random_all[id]
              for id in trial_ind_batch], cpu_offset + batch_ind)
            for batch_ind, trial_ind_batch in enumerate(trial_ind_batch_all))

        with torch.no_grad():
            success_det = []
            success = []
            depth = np.empty((0, self.img_size, self.img_size))
            pred = np.empty((0, 2), dtype='int')

            # executor.submit will not keep the order of calling the function! executor.map will
            # num_cpus=16, 200 envs, fork, 21.6s
            # num_cpus=16, 200 envs, forkserver, 24.5s
            # num_cpus=16, 200 envs, spawn, 24.9s
            # ray take 28.8s
            # 13.8s after batching
            # ? with 32 cpus and 200 envs on server, same when using fork or forkserver
            with concurrent.futures.ProcessPoolExecutor(num_cpus) as executor:
                res_batch_all = list(executor.map(self.sim_step_helper, args))
                for res_batch in res_batch_all:
                    success_det += res_batch[0]
                    success += res_batch[1]
                    depth = np.concatenate((depth, res_batch[2]))
                    pred = np.concatenate((pred, res_batch[3]))
                executor.shutdown()

            # map/starmap will return results in the order of calling the function unlike apply/apply_async; and starmap accepts multiple arguments unlike map
            # with mp.Pool(processes=num_cpus) as pool:
            # 	res_all = pool.starmap(self.sim_step, zip(obj_path_list, add_noise_all))
            # 	for res in res_all:
            # 		success += [res[0]]
            # 		depth = np.concatenate((depth, res[1][np.newaxis]))
            # 		pred = np.concatenate((pred, res[2][np.newaxis]))
            # pool.close()
            # pool.join()

        return success_det, success, depth, pred

    def sim_step_helper(self, args):
        return self.sim_step(args[0], args[1], args[2], args[3])

    def sim_step(self,
                 obj_path_list_all,
                 obj_height_list_all,
                 random_all,
                 cpu_id=0,
                 gui=False):

        # Assign CPU - somehow PyBullet very slow if assigning cpu in GUI mode
        if not gui:
            ps = psutil.Process()
            ps.cpu_affinity([cpu_id])
            torch.set_num_threads(1)

        # Re-seed for sampling initial poses
        np.random.seed()

        # Initialize PyBullet
        if gui:
            p.connect(p.GUI, options="--width=2600 --height=1800")
            p.resetDebugVisualizerCamera(0.8, 180, -45, [0.5, 0, 0])
        else:
            p.connect(p.DIRECT)

        # Params
        initial_ee_pos_before_img = array([0.3, -0.5, 0.25])
        ee_orn = array([1.0, 0.0, 0.0, 0.0])  # straight down

        ######################### Reset #######################
        self.panda_env.reset_env()

        ########################
        success_det_trials = []  # deterministic
        success_trials = []
        depth_trials = np.empty((0, self.img_size, self.img_size))
        pred_trials = np.empty((0, 2), dtype='int')

        for obj_path_list, obj_height_list, use_random_object in zip(
                obj_path_list_all, obj_height_list_all, random_all):

            # If use random, also sample a deterministic one for success rate
            if use_random_object:
                use_random_trial = [1, 0]
            else:
                use_random_trial = [0]
            for use_random in use_random_trial:

                # Set arm to starting pose
                self.panda_env.reset_arm_joints_ik(initial_ee_pos_before_img,
                                                   ee_orn)
                self.panda_env.grasp(targetVel=0.10)  # open gripper

                # At each step, use same environment (objects)
                for obj_id in self.obj_id_list:
                    p.removeBody(obj_id)
                self.load_obj(obj_path_list, obj_height_list)

                # If clears table
                success = 0

                ######################### Execute #######################

                # Infer
                depth = torch.from_numpy(self.get_depth()[np.newaxis]).to(
                    'cpu')  # 1xNxW
                # plt.imshow(depth_orig[0], cmap='Greys', interpolation='nearest')
                # plt.show()
                # for depth in depth_rot_all:
                # 	plt.imshow(depth[0], cmap='Greys', interpolation='nearest')
                # 	plt.show()
                pred_infer = self.policy(depth).squeeze(0).detach().numpy()
                # plt.imshow(pred_infer.detach().cpu().numpy())
                # plt.show()

                # Apply spatial (3D) argmax to pick pixel and theta
                if not use_random:
                    (px, py) = np.unravel_index(np.argmax(pred_infer),
                                                pred_infer.shape)
                else:
                    px = random.randint(0, self.img_size - 1)
                    py = random.randint(0, self.img_size - 1)

                # Get x/y from pixels
                x, y = self.pixel2xy_mat[py, px]  # actual pos, a bug

                # Find the target z height
                z = float(depth[0, px, py] * self.max_obj_height)
                z_target = max(0, z - self.delta_z)  # clip
                z_target_ee = z_target + self.min_ee_z

                # Rotate into local frame
                xy_orig = array([[x], [y]])

                # Execute, reset ik on top of object, reach down, grasp, lift, check success
                ee_pos_before = np.append(xy_orig, z_target_ee + 0.10)
                ee_pos_after = np.append(xy_orig, z_target_ee + 0.05)
                for _ in range(3):
                    self.panda_env.reset_arm_joints_ik(ee_pos_before, ee_orn)
                    p.stepSimulation()
                ee_pos = np.append(xy_orig, z_target_ee)
                self.panda_env.move_pos(ee_pos,
                                        absolute_global_quat=ee_orn,
                                        numSteps=300)
                # print(self.panda_env.get_ee())
                self.panda_env.grasp(targetVel=-0.10)  # always close gripper
                self.panda_env.move_pos(
                    ee_pos, absolute_global_quat=ee_orn,
                    numSteps=100)  # keep pose until gripper closes
                self.panda_env.move_pos(ee_pos_after,
                                        absolute_global_quat=ee_orn,
                                        numSteps=150)  # lift

                # Check if all objects removed, terminate early if so
                self.clear_obj()
                if len(self.obj_id_list) == 0:
                    success = 1

                ######################### Data #######################
                if not (
                        use_random_object and not use_random
                ):  # do not save for the deterministic one paired with the random one
                    success_trials += [success]
                    depth_trials = np.concatenate((depth_trials, depth))
                    pred_trials = np.concatenate(
                        (pred_trials, np.array([[px, py]], dtype='int')))
                if not use_random:  # save success of deterministic one
                    success_det_trials += [success]

        p.disconnect()
        return success_det_trials, success_trials, depth_trials, pred_trials

    def clear_obj(self):
        height = []
        obj_to_be_removed = []
        for obj_id in self.obj_id_list:
            pos, _ = p.getBasePositionAndOrientation(obj_id)
            height += [pos[2]]
            if pos[2] - self.obj_initial_height_list[obj_id] > 0.03:
                obj_to_be_removed += [obj_id]

        for obj_id in obj_to_be_removed:
            p.removeBody(obj_id)
            self.obj_id_list.remove(obj_id)

    def get_depth(self):
        camera_height = 0.30
        viewMat = [
            -1.0, 0.0, -0.0, 0.0, -0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
            0.5, 0.0, -camera_height, 1.0
        ]  # 5cm height
        projMat = [
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
            -1.0000200271606445, -1.0, 0.0, 0.0, -0.02000020071864128, 0.0
        ]
        width = 64
        height = 64
        center = width // 2
        crop_dim = self.img_size
        m22 = projMat[10]
        m32 = projMat[14]
        near = 2 * m32 / (2 * m22 - 2)
        far = ((m22 - 1.0) * near) / (m22 + 1.0)

        img_arr = p.getCameraImage(width=width,
                                   height=height,
                                   viewMatrix=viewMat,
                                   projectionMatrix=projMat,
                                   flags=p.ER_NO_SEGMENTATION_MASK)
        depth = np.reshape(
            img_arr[3],
            (width, height))[center - crop_dim // 2:center + crop_dim // 2,
                             center - crop_dim // 2:center + crop_dim // 2]
        depth = far * near / (far - (far - near) * depth)

        depth = (camera_height -
                 depth) / self.max_obj_height  # set table zero, and normalize
        depth = depth.clip(min=0., max=1.)

        return depth


class TrainGrasp:
    def __init__(self,
                 result_dir,
                 num_cpus=16,
                 cpu_offset=0,
                 device='cuda:0',
                 img_size=10,
                 batch_size=64,
                 buffer_size=500,
                 hidden_size=200,
                 lr=1e-4,
                 num_update_per_step=1,
                 eps=0.2,
                 eps_min=0.,
                 eps_decay=0.9,
                 **kwargs):
        # Save class attributes and initialize folders
        save__init__args(locals())
        self.model_dir = result_dir + 'policy_model/'
        self.img_dir = result_dir + 'policy_img/'
        self.train_detail_dir = result_dir + 'retrain_detail/'
        ensure_directory(self.model_dir)
        ensure_directory(self.img_dir)
        ensure_directory(self.train_detail_dir)

        # Using CPU for inference in simulation right now
        self.graspSim = GraspSim(
            img_size=img_size,
            hidden_size=hidden_size,
        )

        # Set up model
        self.policy = MLP(hidden_size=self.hidden_size,
                          img_size=self.img_size).to(device)
        num_model_parameter = sum(p.numel() for p in self.policy.parameters()
                                  if p.requires_grad)
        print('Num of policy parameters: %d' % num_model_parameter)

        # Optimizer
        self.criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')
        self.optimizer = torch.optim.AdamW([{
            'params': self.policy.parameters(),
            'lr': lr,
            'weight_decay': 0
        }])

        # Flag for indicating first iteration
        self.initial_itr_flag = True

        # Reset buffer
        self.reset_buffer()

    def reset_buffer(self):
        # Experience buffer
        self.depth_buffer = torch.empty(
            (0, self.img_size, self.img_size)).float().to('cpu')
        self.ground_truth_buffer = torch.empty(
            (0, self.img_size, self.img_size)).float().to('cpu')
        self.mask_buffer = torch.empty(
            (0, self.img_size, self.img_size)).float().to('cpu')
        self.recency_buffer = np.empty((0))

    def run(self,
            obj_path_all,
            obj_height_all,
            num_step,
            num_trial_per_step=10,
            debug_freq=100,
            affordance_map_freq=100,
            **kwargs):

        # Record results
        train_loss_list = []
        success_rate_list = []
        best_success_rate = 0.
        best_policy_path = None
        prev_policy_path = None

        # Start with all random to fill up the buffer
        if self.initial_itr_flag:
            eps = 1
            self.initial_itr_flag = False
        else:
            eps = self.eps

        # Simulate once first
        new_success_det, new_success, new_depth, new_pred = self.graspSim.sim_parallel(
            obj_path_all * num_trial_per_step,
            obj_height_all * num_trial_per_step,
            eps=eps,
            num_cpus=self.num_cpus,
            cpu_offset=self.cpu_offset)

        # Run
        for step in range(num_step):
            # optimizer_to(self.optimizer, 'cpu')# had to push optimizer to cpu
            # optimizer_to(self.optimizer, self.device)	# push back to gpu

            # Decay epsilon if buffer filled
            eps = max(self.eps_min, eps * self.eps_decay)

            # Add to buffer
            fill_flag = self.add_to_buffer(step, new_success, new_depth,
                                           new_pred)

            # Update for multiple times once buffer filled
            step_loss = 0
            if fill_flag:
                for _ in range(self.num_update_per_step):
                    step_loss += self.train_policy()
                step_loss /= self.num_update_per_step

            # Update policy for graspSim (on CPU)
            self.graspSim.update_policy(self.get_policy())

            # Simulate
            new_success_det, new_success, new_depth, new_pred = self.graspSim.sim_parallel(
                obj_path_all * num_trial_per_step,
                obj_height_all * num_trial_per_step,
                eps=eps,
                num_cpus=self.num_cpus,
                cpu_offset=self.cpu_offset)
            success_rate = np.mean(array(new_success_det))

            # Record
            train_loss_list += [step_loss]
            success_rate_list += [success_rate]

            # Generate sample affordance map - samples can be random - so not necessarily the best one
            if step % affordance_map_freq == 0:
                depth_ind = random.randint(0, new_depth.shape[0] - 1)
                depth_input = torch.from_numpy(
                    new_depth[depth_ind][np.newaxis]).float().to(
                        self.device)  # 1xHxW
                pred_infer = self.policy(depth_input).squeeze(0)  # HxW
                self.save_infer_img(new_depth[depth_ind],
                                    pred_infer,
                                    img_path_prefix=self.img_dir + str(step) +
                                    '_' + str(depth_ind))

            # Debug
            if step % debug_freq == 0:
                print("Step {:d}, Loss: {:.4f}".format(step, step_loss))
                torch.save(
                    {
                        'train_loss_list': train_loss_list,
                        'success_rate_list': success_rate_list,
                    }, self.result_dir + 'train_details')  # keeps overwriting
                # Clear GPU data regularly
                torch.cuda.empty_cache()

            # Save model if better success rate, remove prev one
            if best_success_rate < success_rate:
                best_success_rate = success_rate
                best_policy_path = self.model_dir + 'step_' + str(
                    step) + '_acc_' + "{:.3f}".format(success_rate)
                self.save_model(path=best_policy_path)
                print('Saving new model, success %f' % success_rate)

                if prev_policy_path is not None:
                    os.remove(prev_policy_path + '.pt')
                prev_policy_path = best_policy_path
        return best_policy_path

    def add_to_buffer(self, step, new_success, new_depth, new_pred):
        # Indices to be replaced in the buffer for current step
        num_new = new_depth.shape[0]

        # Convert depth to tensor and append new dimension
        new_depth = torch.from_numpy(new_depth).float().to('cpu').detach()

        # Construnct ground truth and mask (all zeros except for selected pixel)
        new_ground_truth = torch.zeros(num_new, self.img_size,
                                       self.img_size).to('cpu')
        new_mask = torch.zeros(num_new, self.img_size, self.img_size).to('cpu')
        for trial_ind, (success,
                        (px, py)) in enumerate(zip(new_success, new_pred)):
            new_ground_truth[trial_ind, px, py] = success
            new_mask[trial_ind, px, py] = 1

        # Determine recency for new data
        recency = math.exp(-step * 0.1)  # rank-based

        # Check if buffer filled up
        if self.depth_buffer.shape[0] < self.buffer_size:
            self.depth_buffer = torch.cat(
                (self.depth_buffer, new_depth))[:self.buffer_size]
            self.ground_truth_buffer = torch.cat(
                (self.ground_truth_buffer,
                 new_ground_truth))[:self.buffer_size]
            self.mask_buffer = torch.cat(
                (self.mask_buffer, new_mask))[:self.buffer_size]
            self.recency_buffer = np.concatenate(
                (self.recency_buffer, np.ones(
                    (num_new)) * recency))[:self.buffer_size]
        else:
            # Replace older ones
            replace_ind = np.random.choice(self.buffer_size,
                                           size=num_new,
                                           replace=False,
                                           p=self.recency_buffer /
                                           np.sum(self.recency_buffer))
            self.depth_buffer[replace_ind] = new_depth
            self.ground_truth_buffer[replace_ind] = new_ground_truth
            self.mask_buffer[replace_ind] = new_mask
            self.recency_buffer[replace_ind] = recency

        # Return if filled up
        if self.depth_buffer.shape[0] >= self.buffer_size:
            return 1
        else:
            return 0

    def train_policy(self):
        # Switch mode
        # self.fcn.to(self.device)
        # self.optimizer.load_state_dict(self.optimizer.state_dict())
        # self.fcn.train()

        # Train by sampling from buffer
        sample_inds = random.sample(range(self.buffer_size), k=self.batch_size)
        depth_train_batch = self.depth_buffer[sample_inds].clone().detach().to(
            self.device, non_blocking=True)  # NxHxW
        ground_truth_batch = self.ground_truth_buffer[sample_inds].clone(
        ).detach().to(self.device, non_blocking=True)  # NxHxW
        mask_train_batch = self.mask_buffer[sample_inds].clone().detach().to(
            self.device, non_blocking=True)  # NxHxW

        # Forward, get loss, zero gradients
        pred_train_batch = self.policy(depth_train_batch)  # NxHxW
        train_loss = self.criterion(pred_train_batch, ground_truth_batch)
        self.optimizer.zero_grad()

        # mask gradient for non-selected pixels
        pred_train_batch.retain_grad()
        pred_train_batch.register_hook(lambda grad: grad * mask_train_batch)

        # Update params using clipped gradients
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 10)
        self.optimizer.step()
        return train_loss.detach().cpu().numpy()

    def load_policy(self, policy_path):
        self.policy.load_state_dict(
            torch.load(policy_path + '.pt', map_location=self.device))
        self.graspSim.update_policy(self.get_policy())

    def get_policy(self):
        return self.policy.state_dict()

    def save_model(self, path):
        torch.save(self.policy.state_dict(), path + '.pt')

    def save_infer_img(self, depth, pred_infer, img_path_prefix):
        depth_8bit = (depth * 255).astype('uint8')
        depth_8bit = np.stack((depth_8bit, ) * 3, axis=-1)
        img_rgb = Image.fromarray(depth_8bit, mode='RGB')
        img_rgb.save(img_path_prefix + '_rgb.png')

        cmap = plt.get_cmap('jet')
        pred_infer_detach = (torch.sigmoid(pred_infer)).detach().cpu().numpy()
        pred_infer_detach = (pred_infer_detach - np.min(pred_infer_detach)) / (
            np.max(pred_infer_detach) - np.min(pred_infer_detach))  # normalize
        pred_cmap = cmap(pred_infer_detach)
        pred_cmap = (np.delete(pred_cmap, 3, 2) * 255).astype('uint8')
        img_heat = Image.fromarray(pred_cmap, mode='RGB')
        img_heat.save(img_path_prefix + '_heatmap.png')

        img_overlay = Image.blend(img_rgb, img_heat, alpha=.8)
        img_overlay.save(img_path_prefix + '_overlay.png')


if __name__ == '__main__':

    # Fix seeds
    # seed = 0
    # random.seed(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)

    # Configs
    name = 'grasp_test'
    num_cpus = 10  # same as num_trial_per_step
    num_steps = 1000
    num_trial_per_step = 10
    num_update_per_step = 10
    batch_size = 16
    val_freq = 20
    buffer_size = 200
    img_size = 20
    result_dir = 'result/' + name + '/'
    ensure_directory(result_dir)

    # Configure objects
    # obj_dir = '/home/allen/data/wasserstein/grasp/random_polygon_v1/'
    # obj_path_all = [obj_dir + str(ind) + '.urdf' for ind in range(10)]
    obj_path_all = ['data/sample_mug/4.urdf']
    obj_height_all = [0.05 for _ in range(len(obj_path_all))]

    # Initialize trianing env
    trainer = TrainGrasp(result_dir=result_dir,
                         buffer_size=buffer_size,
                         num_update_per_step=num_update_per_step,
                         batch_size=batch_size,
                         img_size=img_size,
                         policy_path=None,
                         num_cpus=num_cpus)
    best_policy_path = trainer.run(obj_path_all, obj_height_all, num_steps,
                                   num_trial_per_step, val_freq, val_freq)
    print('Training done; best policy path: ', best_policy_path)
