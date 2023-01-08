import os
import numpy as np
import random
import torch
import logging
import wandb
from copy import deepcopy
from PIL import Image
import matplotlib.pyplot as plt

from agent.agent_grasp import AgentGrasp
from util.image import rotate_tensor

from util.scheduler import StepLRFixed


class AgentGraspEq(AgentGrasp):
    def __init__(self, cfg, venv, verbose=True):
        """
        """
        super().__init__(cfg, venv, verbose)


    def store_cfg(self, cfg):
        
        self.cfg_action_decoder_training = cfg.action_decoder_training
        self.target_reward = cfg.target_reward
        self.action_dim = 2

        # self.img_h = cfg.learner.img_h
        # self.img_w = cfg.learner.img_w
        # self.max_sample_steps = cfg.max_sample_steps
        # self.batch_size = cfg.batch_size
        # self.memory_capacity = cfg.memory_capacity
        # self.update_freq = cfg.update_freq
        # self.num_update = max(1, int(cfg.replay_ratio*self.update_freq/self.batch_size))
        # self.check_freq = cfg.check_freq
        # self.num_warmup_step_percentage = cfg.num_warmup_step_percentage
        # self.num_episode_per_eval = cfg.num_eval_episode
        # self.cfg_eps = cfg.eps
        # self.num_affordance = cfg.num_affordance


    # TODO: move to base class
    def learn(self, tasks=None, 
                    memory=None,
                    policy_path=None, 
                    optimizer_state=None,
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

        # flag for checking whether training the action decoder or training the latent policy plus the state encoder - initially we train the action decoder
        flag_train_action_decoder = True 

        # Run rest of steps while optimizing policy
        cnt_opt = 0
        best_reward = 0
        while self.cnt_step <= self.max_sample_steps:
            print(self.cnt_step, end='\r')

            # Train
            if not self.eval_mode:

                # Run steps
                self.cnt_step += self.run_steps(num_step=self.update_freq)[0]

                # Update either action decoder or latent policy plus state encoder
                loss = np.empty((0,3))
                for _ in range(self.num_update):
                    batch_train = self.unpack_batch(self.sample_batch())
                    if flag_train_action_decoder:
                        action_decoder_ce_loss = self.learner.update_action_decoder(batch_train)
                        latent_policy_ce_loss = 0
                        alignment_loss = 0 
                    else:
                        latent_policy_ce_loss, alignment_loss = self.learner.update_latent_policy(batch_train)
                        action_decoder_ce_loss = 0
                    loss = np.vstack((loss, [action_decoder_ce_loss, latent_policy_ce_loss, alignment_loss]))
                loss = np.sum(loss, axis=1) / self.num_update

                # Record:
                self.loss_record[self.cnt_step] = loss
                if self.use_wandb:
                    wandb.log(
                        {
                            "Train loss": loss,
                        },
                        step=self.cnt_step,
                        commit=False)

                # Count number of optimization
                cnt_opt += 1

                # Clear GPU cache
                torch.cuda.empty_cache()

                # Evaluate
                if cnt_opt % self.check_freq == 0 and cnt_opt > 0:
                    self.set_eval_mode()

            # Evaluate
            else:
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

                # Check if target reward is reached
                flag_train_action_decoder = eval_reward_cumulative < self.target_reward

                # Saving model (and replay buffer)
                if self.save_metric == 'cum_reward':
                    best_path = self.save(metric=eval_reward_cumulative)
                else:
                    raise NotImplementedError

                # Generate sample affordance map - samples can be random - so not necessarily the best one
                # self.save_sample_affordance(num=self.num_affordance)

                # Save training details
                torch.save(
                    {
                        'loss_record': self.loss_record,
                        'eval_record': self.eval_record,
                    }, os.path.join(self.out_folder, 'train_details'))

                # Switch to training
                self.set_train_mode()

        ################### Done ###################
        # best_path = self.save(force_save=True)    # TODO: force_save cfg
        best_reward = np.max([q[0] for q in self.pq_top_k.queue]) # yikes...
        logging.info('Saving best path {} with reward {}!'.format(best_path, best_reward))

        # Policy, memory, optimizer
        return best_path, deepcopy(self.memory), self.learner.get_optimizer_state()


    # === Replay and update ===
    def sample_batch(self, batch_size=None):
        # Sample indices
        if batch_size is None:
            batch_size = self.batch_size
        buffer_size = self.depth_buffer.shape[0]
        sample_inds = random.sample(range(buffer_size), k=batch_size)
        
        # Get obs with indices
        depth_batch = self.depth_buffer[sample_inds].clone().detach().to(
            self.device, non_blocking=True).unsqueeze(1)  # Nx1xHxW
        ground_truth_batch = self.ground_truth_buffer[sample_inds].clone(
        ).detach().to(self.device, non_blocking=True)  # NxHxW
        mask_batch = self.mask_buffer[sample_inds].clone().detach().to(
            self.device, non_blocking=True)  # NxHxW
        
        # Get action with indices
        action_batch = self.action_buffer[sample_inds].clone().detach().to(
            self.device, non_blocking=True)  # Nx2
        
        return (depth_batch, ground_truth_batch, mask_batch, action_batch)


    def store_transition(self, s, a, r, s_, done, info):
        """Different from typical RL buffer setup"""

        # Indices to be replaced in the buffer for current step
        num_new = s.shape[0]
        assert num_new == 1

        # Extract action
        _, _, _, theta, py, px = a
        action_tensor = torch.tensor([py, px]).unsqueeze(0)

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
            self.action_buffer = torch.cat(
                (self.mask_buffer, action_tensor))[:self.memory_capacity]
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
            self.action_buffer[replace_ind] = action_tensor


    def reset_memory(self, memory):
        """Also building action buffer"""
        super().reset_memory(memory)

        self.action_buffer = torch.empty((0, self.action_dim)).float().to('cpu')
        logging.info('Built action buffer!')
