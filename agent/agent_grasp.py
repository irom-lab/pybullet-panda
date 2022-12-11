import os
import numpy as np
import random
import torch
import logging
import wandb
from copy import deepcopy
from PIL import Image
import matplotlib.pyplot as plt

from agent.agent_base import AgentBase
from agent.learner import get_learner
from util.image import rotate_tensor

from alano.train.scheduler import StepLRFixed


class AgentGrasp(AgentBase):
    def __init__(self, cfg, venv, verbose=True):
        """
        """
        super().__init__(cfg, venv)
        self.store_cfg(cfg)

        # Learner
        self.learner_name = cfg.learner.name
        self.learner = get_learner(self.learner_name)(cfg.learner)
        self.learner.build_network(cfg.learner.arch, verbose=verbose)
        self.module_all = [self.learner]    # for saving models

        # Utility - helper functions for envs
        # self.utility = get_utility(cfg.utility.name)(cfg.utility)

        # Affordance map
        self.img_folder = os.path.join(cfg.out_folder, 'img')
        os.makedirs(self.img_folder, exist_ok=True)


    def store_cfg(self, cfg):
        self.img_h = cfg.learner.img_h
        self.img_w = cfg.learner.img_w
        self.max_sample_steps = cfg.max_sample_steps
        self.batch_size = cfg.batch_size
        self.memory_capacity = cfg.memory_capacity
        self.update_freq = cfg.update_freq
        self.num_update = max(1, int(cfg.replay_ratio*self.update_freq/self.batch_size))
        self.check_freq = cfg.check_freq
        self.num_warmup_step_percentage = cfg.num_warmup_step_percentage
        self.num_episode_per_eval = cfg.num_eval_episode
        self.cfg_eps = cfg.eps
        self.num_affordance = cfg.num_affordance


    # TODO: move to base class
    def learn(self, tasks=None, 
                    memory=None,
                    policy_path=None, 
                    optimizer_state=None,
                    # out_folder_postfix=None,
                    verbose=False,
                    **kwargs):
        logging.info('Learning with {} steps!'.format(self.max_sample_steps))
        self.cnt_step = 0
        self.reset_save_info(self.out_folder)

        # Reset tasks
        if tasks is not None:
            self.reset_tasks(tasks, verbose)

        # Set up memory
        self.reset_memory(memory)

        # Set up policy
        self.reset_policy(policy_path)

        # Set up optimizer
        self.reset_optimizer(optimizer_state)

        # Exploration
        if hasattr(self.cfg_eps, 'period_percentage'):
            eps_period = int(self.max_sample_steps*self.cfg_eps.period_percentage)
        else:
            eps_period = self.cfg_eps.period
        self.eps_schduler = StepLRFixed(initValue=self.cfg_eps.init,
                                        period=eps_period,
                                        endValue=self.cfg_eps.end,
                                        stepSize=self.cfg_eps.step)

        # Run initial steps
        self.set_train_mode()
        if self.num_warmup_step_percentage > 0:
            num_warmup_step = int(self.max_sample_steps*self.num_warmup_step_percentage)
            self.cnt_step, _ = self.run_steps(num_step=num_warmup_step)
        logging.info(f'Warmed up with {self.cnt_step} steps!')

        # Run rest of steps while optimizing policy
        cnt_opt = 0
        best_reward = 0
        while self.cnt_step <= self.max_sample_steps:
            print(self.cnt_step, end='\r')

            # Train or eval
            if self.eval_mode:
                num_episode_run, _ = self.run_steps(num_episode=self.num_episode_per_eval)
                eval_reward_cumulative = self.eval_reward_cumulative_all / num_episode_run
                if verbose:
                    logging.info(f'eps: {self.eps_schduler.get_variable()}')
                    logging.info(f'avg cumulative reward: {eval_reward_cumulative}')
                self.eval_record[self.cnt_step] = (eval_reward_cumulative, )
                if self.use_wandb:
                    wandb.log({
                        "Cumulative Reward": eval_reward_cumulative,
                    }, step=self.cnt_step, commit=True)

                # Saving model (and replay buffer)
                if self.save_metric == 'cum_reward':
                    best_path = self.save(metric=eval_reward_cumulative)
                else:
                    raise NotImplementedError

                # Generate sample affordance map - samples can be random - so not necessarily the best one
                self.save_sample_affordance(num=self.num_affordance)

                # Save training details
                torch.save(
                    {
                        'loss_record': self.loss_record,
                        'eval_record': self.eval_record,
                    }, os.path.join(self.out_folder, 'train_details'))

                # Switch to training
                self.set_train_mode()
            else:
                self.cnt_step += self.run_steps(num_step=self.update_freq)[0]

                # Update policy
                loss = 0
                for _ in range(self.num_update):
                    batch_train = self.unpack_batch(self.sample_batch())
                    loss_batch = self.learner.update(batch_train)
                    loss += loss_batch
                loss /= self.num_update

                # Record: loss_q, loss_pi, loss_entropy, loss_alpha
                self.loss_record[self.cnt_step] = loss
                if self.use_wandb:
                    wandb.log(
                        {
                            "Train loss": loss,
                        },
                        step=self.cnt_step,
                        commit=False)

                # Reset simulation
                # self.reset_sim()

                # Count number of optimization
                cnt_opt += 1

                # Clear GPU cache
                torch.cuda.empty_cache()

                ################### Eval ###################
                if cnt_opt % self.check_freq == 0:
                    self.set_eval_mode()

        ################### Done ###################
        # best_path = self.save(force_save=True)    # TODO: force_save cfg
        best_reward = np.max([q[0] for q in self.pq_top_k.queue]) # yikes...
        logging.info('Saving best path {} with reward {}!'.format(best_path, best_reward))

        # Policy, memory, optimizer
        return best_path, deepcopy(self.memory), self.learner.get_optimizer_state()


    # === Replay and update ===
    def sample_batch(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size

        # Train by sampling from buffer
        buffer_size = self.depth_buffer.shape[0]
        sample_inds = random.sample(range(buffer_size), k=batch_size)
        depth_train_batch = self.depth_buffer[sample_inds].clone().detach().to(
            self.device, non_blocking=True).unsqueeze(1)  # Nx1xHxW
        ground_truth_batch = self.ground_truth_buffer[sample_inds].clone(
        ).detach().to(self.device, non_blocking=True)  # NxHxW
        mask_train_batch = self.mask_buffer[sample_inds].clone().detach().to(
            self.device, non_blocking=True)  # NxHxW
        return (depth_train_batch, ground_truth_batch, mask_train_batch)


    def unpack_batch(self, batch):
        return batch


    def store_transition(self, s, a, r, s_, done, info):
        """Different from typical RL buffer setup"""
        # TODO: batch store?

        # Indices to be replaced in the buffer for current step
        num_new = s.shape[0]
        assert num_new == 1

        # Extract action
        _, _, _, theta, py, px = a

        # Convert depth to tensor
        new_depth = s.detach().to('cpu')

        # Rotate according to theta
        new_depth = rotate_tensor(new_depth, theta=torch.tensor(theta)).squeeze(1)

        # Construnct ground truth and mask (all zeros except for selected pixel)
        new_ground_truth = torch.zeros_like(new_depth).to('cpu')
        new_mask = torch.zeros_like(new_depth).to('cpu')
        # for trial_ind, (success, (py, px)) in enumerate(zip(r, a)):
        #     new_ground_truth[trial_ind, py, py] = success
        #     new_mask[trial_ind, py, px] = 1
        new_ground_truth[0, int(py), int(px)] = r
        new_mask[0, int(py), int(px)] = 1

        # Determine recency for new data
        # recency = np.exp(-self.cnt_step * 0.1)  # rank-based    #?

        # Check if buffer filled up
        if self.depth_buffer.shape[0] < self.memory_capacity:
            self.depth_buffer = torch.cat(
                (self.depth_buffer, new_depth))[:self.memory_capacity]
            self.ground_truth_buffer = torch.cat(
                (self.ground_truth_buffer,
                 new_ground_truth))[:self.memory_capacity]
            self.mask_buffer = torch.cat(
                (self.mask_buffer, new_mask))[:self.memory_capacity]
            # self.recency_buffer = np.concatenate(
            #     (self.recency_buffer, np.ones(
            #         (num_new)) * recency))[:self.memory_capacity]
        else:
            # Replace older ones
            replace_ind = np.random.choice(self.memory_capacity,
                                           size=num_new,
                                           replace=False,
                                        #    p=self.recency_buffer /
                                        #    np.sum(self.recency_buffer)
                                           )
            self.depth_buffer[replace_ind] = new_depth
            self.ground_truth_buffer[replace_ind] = new_ground_truth
            self.mask_buffer[replace_ind] = new_mask
            # self.recency_buffer[replace_ind] = recency


    #== Reset policy/optimizer/memory
    def reset_policy(self, policy_path=None):
        if policy_path:
            self.learner.load_network(policy_path)
            logging.info('Loaded policy network from: {}'.format(policy_path))
        else:
            self.learner.build_network(self.cfg.learner.arch,   
                                       build_optimizer=False, 
                                       verbose=False)
            logging.info('Built new policy network!')


    def reset_memory(self, memory):
        if memory is not None:
            raise NotImplementedError
            # self.memory = memory
            # logging.info('Reusing memory with size {}!'.format(len(self.memory)))
        elif hasattr(self, 'memory_path'):
            raise NotImplementedError
        else:
            self.depth_buffer = torch.empty(
                (0, self.img_h, self.img_w)).float().to('cpu')
            self.ground_truth_buffer = torch.empty(
                (0, self.img_h, self.img_w)).float().to('cpu')
            self.mask_buffer = torch.empty(
                (0, self.img_h, self.img_w)).float().to('cpu')
            # self.recency_buffer = np.empty((0))
            logging.info('Built memory!')


    def reset_optimizer(self, optimizer_state=None):
        if optimizer_state:
            self.learner.load_optimizer_state(optimizer_state)
            logging.info('Loaded policy optimizer!')
        else:
            self.learner.build_optimizer()
            logging.info('Built new policy optimizer!')


    def save_sample_affordance(self, num):
        for ind in range(num):
            img_ind = self.rng.integers(0, self.depth_buffer.shape[0])
            img = self.depth_buffer[img_ind]
            img_input = img[np.newaxis, np.newaxis].float().to(self.device)  # 1x1xHxW
            pred = self.learner(img_input).squeeze(1).squeeze(0)  # HxW
            img_path_prefix = os.path.join(self.img_folder, str(self.cnt_step))

            depth_8bit = (img.detach().cpu().numpy() * 255).astype('uint8')
            depth_8bit = np.stack((depth_8bit, ) * 3, axis=-1)
            img_rgb = Image.fromarray(depth_8bit, mode='RGB')
            img_rgb.save(img_path_prefix + f'_{ind}_rgb.png')

            cmap = plt.get_cmap('jet')
            pred_detach = (torch.sigmoid(pred)).detach().cpu().numpy()
            pred_detach = (pred_detach - np.min(pred_detach)) / (
                np.max(pred_detach) - np.min(pred_detach))  # normalize
            pred_cmap = cmap(pred_detach)
            pred_cmap = (np.delete(pred_cmap, 3, 2) * 255).astype('uint8')
            img_heat = Image.fromarray(pred_cmap, mode='RGB')
            img_heat.save(img_path_prefix + f'_{ind}_heatmap.png')

            img_overlay = Image.blend(img_rgb, img_heat, alpha=.5)
            img_overlay.save(img_path_prefix + f'_{ind}_overlay.png')
