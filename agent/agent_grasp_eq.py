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
        super().store_cfg(cfg)
        self.target_reward = cfg.target_reward
        self.action_dim = 2


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

        # Run initial steps with random steps
        self.set_train_mode()
        if self.num_warmup_step_percentage > 0:
            num_warmup_step = int(self.max_sample_steps*self.num_warmup_step_percentage)
            self.cnt_step, _ = self.run_steps(num_step=num_warmup_step,
                                              force_random=True)
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
                for update_ind in range(self.num_update):
                    if flag_train_action_decoder:
                        batch_train = self.unpack_batch(self.sample_batch())
                        action_decoder_ce_loss = self.learner.update_action_decoder(batch_train, verbose=update_ind < 1)
                        # latent_policy_ce_loss = 0
                        # alignment_loss = 0 
                    else:
                        action_decoder_ce_loss = 0

                    batch_train = self.unpack_batch(self.sample_batch(positive_ratio=self.batch_positive_ratio))
                    latent_policy_ce_loss, alignment_loss, stats = self.learner.update_latent_policy(batch_train, verbose=update_ind < 1)
                    loss = np.vstack((loss, [action_decoder_ce_loss, latent_policy_ce_loss, alignment_loss]))
                loss = np.sum(loss, axis=1) / self.num_update

                # Record:
                self.loss_record[self.cnt_step] = loss
                if self.use_wandb:
                    wandb.log(
                        {
                            "CE loss from action decoder": loss[0],
                            "CE loss from latent policy": loss[1],
                            "Alignment loss": loss[2],
                        },
                        step=self.cnt_step,
                        commit=False)

                # Count number of optimization
                cnt_opt += 1

                # Clear GPU cache
                torch.cuda.empty_cache()

                # Switch to evaluation
                if cnt_opt % self.check_freq == 0 and cnt_opt > 0:
                    self.set_eval_mode()

            # Evaluate
            else:
                logging.info("======================================")
                logging.info(f'Evaluating at step {self.cnt_step}...')
                num_episode_run, _ = self.run_steps(num_episode=self.num_episode_per_eval)
                eval_reward_cumulative = self.eval_reward_cumulative_all / num_episode_run
                if verbose:
                    logging.info(f'eps: {self.eps_schduler.get_variable()}')
                    logging.info(f'avg cumulative reward: {eval_reward_cumulative}')
                    logging.info(f'number of positive examples in the buffer: {len(torch.where(self.reward_buffer == 1)[0])}')
                self.eval_record[self.cnt_step] = (eval_reward_cumulative, )
                if self.use_wandb:
                    wandb.log({
                        "Cumulative Reward": eval_reward_cumulative,
                    }, step=self.cnt_step, commit=True)

                # Check if target reward is reached
                flag_train_action_decoder = eval_reward_cumulative < self.target_reward
                logging.info(f'Training action decoder next? {flag_train_action_decoder}')

                # Saving model (and replay buffer)
                if self.save_metric == 'cum_reward':
                    best_path = self.save(metric=eval_reward_cumulative)
                else:
                    raise NotImplementedError

                #!
                self.eps_schduler = StepLRFixed(initValue=1-eval_reward_cumulative,
                                                period=eps_period,
                                                endValue=self.cfg_eps.end,
                                                stepSize=self.cfg_eps.step)

                # Generate sample affordance map - samples can be random - so not necessarily the best one
                with torch.no_grad():
                    self.save_sample_affordance(num=self.num_affordance)

                # Save training details
                torch.save(
                    {
                        'loss_record': self.loss_record,
                        'eval_record': self.eval_record,
                    }, os.path.join(self.out_folder, 'train_details'))

                # Switch to training
                self.set_train_mode()
                logging.info("======================================")

        ################### Done ###################
        # best_path = self.save(force_save=True)    # TODO: force_save cfg
        best_reward = np.max([q[0] for q in self.pq_top_k.queue]) # yikes...
        logging.info('Saving best path {} with reward {}!'.format(best_path, best_reward))

        # Policy
        return best_path


    # === Replay and update ===
    def sample_batch(self, batch_size=None, positive_ratio=None):
        # Sample indices
        if batch_size is None:
            batch_size = self.batch_size

        # Train by sampling from buffer
        buffer_size = self.depth_buffer.shape[0]
        if positive_ratio is None:
            sample_inds = random.sample(range(buffer_size), k=batch_size)
        else:
            positive_inds = torch.where(self.reward_buffer == 1)[0]
            negative_inds = torch.where(self.reward_buffer == 0)[0]

            #! Do not sample more than 20% of positive examples in each batch, while also keeping a minimum amount
            num_positive = min(int(len(positive_inds)*0.2), int(batch_size * positive_ratio))
            num_positive = num_positive if num_positive > 16 else 0

            positive_sample_inds = positive_inds[random.sample(range(len(positive_inds)), 
                                                        k=min(num_positive, len(positive_inds)))]
            num_negative = batch_size - len(positive_sample_inds)
            negative_sample_inds = negative_inds[random.sample(range(len(negative_inds)), 
                                                               k=num_negative)]
            sample_inds = torch.cat((positive_sample_inds, negative_sample_inds))  
            sample_inds = sample_inds[torch.randperm(len(sample_inds))]

        # Get obs with indices
        depth_batch = self.depth_buffer[sample_inds].clone().detach().to(
            self.device, non_blocking=True).unsqueeze(1)  # Nx1xHxW
        ground_truth_batch = self.ground_truth_buffer[sample_inds].clone(
        ).detach().to(self.device, non_blocking=True)  # NxHxW
        mask_batch = self.mask_buffer[sample_inds].clone().detach().to(
            self.device, non_blocking=True)  # NxHxW
        
        # Get action and reward with indices
        action_batch = self.action_buffer[sample_inds].clone().detach().to(
            self.device, non_blocking=True)  # Nx2
        reward_batch = self.reward_buffer[sample_inds].clone().detach().to(
            self.device, non_blocking=True)  # Nx1
        scaling_batch = self.scaling_buffer[sample_inds].clone().detach().to(
            self.device, non_blocking=True)  # Nx1
        
        return (depth_batch, ground_truth_batch, mask_batch, action_batch, reward_batch, scaling_batch)


    def store_transition(self, s, a, r, s_, done, info):
        """Different from typical RL buffer setup"""

        # Indices to be replaced in the buffer for current step
        num_new = s.shape[0]
        assert num_new == 1

        # Extract action
        _, _, _, theta, py, px = a
        action_tensor = torch.tensor([[py, px]]).float()
        reward_tensor = torch.tensor([r]).float()
        scaling_tensor = torch.tensor([info['global_scaling']]).float()

        # Convert depth to tensor
        new_depth = s.detach().to('cpu')

        # Rotate according to theta
        new_depth = rotate_tensor(new_depth, theta=torch.tensor(theta)).squeeze(1)

        # Construnct ground truth and mask (all zeros except for selected pixel)
        new_ground_truth = torch.zeros_like(new_depth).to('cpu')
        new_mask = torch.zeros_like(new_depth).to('cpu')
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
                (self.action_buffer, action_tensor))[:self.memory_capacity]
            self.reward_buffer = torch.cat(
                (self.reward_buffer, reward_tensor))[:self.memory_capacity]
            self.scaling_buffer = torch.cat(
                (self.scaling_buffer, scaling_tensor))[:self.memory_capacity]
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
            self.reward_buffer[replace_ind] = reward_tensor
            self.scaling_buffer[replace_ind] = scaling_tensor


    def reset_memory(self, memory):
        """Also building action buffer"""
        super().reset_memory(memory)

        self.action_buffer = torch.empty((0, self.action_dim)).float().to('cpu')
        self.scaling_buffer = torch.empty((0)).float().float().to('cpu')
        logging.info('Built action and reward buffer!')


    def save_sample_affordance(self, num):
        for ind in range(num):
            img_ind = self.rng.integers(0, self.depth_buffer.shape[0])
            img = self.depth_buffer[img_ind]
            scaling = self.scaling_buffer[img_ind]
            img_input = img[np.newaxis, np.newaxis].float().to(self.device)  # 1x1xHxW
            pred = self.learner(img_input).squeeze(1).squeeze(0)  # HxW
            img_path_prefix = os.path.join(self.img_folder, f'step-{str(self.cnt_step)}_scaling-{scaling:.3f}')

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
