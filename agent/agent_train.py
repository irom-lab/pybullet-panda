import os
import numpy as np
import random
import torch
import logging
import wandb

from agent.agent_base import AgentBase
from agent.learner import get_learner
from agent.utility import get_utility
from util.image import rotate_tensor
from util.scheduler import StepLRFixed
from util.image import save_affordance_map


class AgentTrain(AgentBase):
    def __init__(self, cfg, venv, verbose=True):
        """
        Run policy training while collecting experiences from environments.
        """
        super().__init__(cfg, venv)
        self.store_cfg(cfg)

        # Learner
        self.learner_name = cfg.learner.name
        self.learner = get_learner(self.learner_name)(cfg.learner)
        # self.learner.build_network(cfg.learner.arch, verbose=verbose)
        self.module_all = [self.learner]    # for saving models

        # Utility - helper functions for envs
        self.utility = get_utility(cfg.utility.name)(cfg.utility)

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
        self.num_epi_per_eval = cfg.num_eval_episode
        self.cfg_eps = cfg.eps
        self.num_affordance = cfg.num_affordance
        self.num_obs_channel = cfg.num_obs_channel


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

        # Run rest of steps while optimizing policy
        cnt_opt = 0
        best_reward = 0
        while self.cnt_step <= self.max_sample_steps:
            print(self.cnt_step, end='\r')

            # Train 
            if not self.eval_mode:

                # Run steps
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
                            "AgentTrain - loss": loss,
                        }, step=self.cnt_step, commit=False)

                # Count number of optimization
                cnt_opt += 1

                # Switch to evaluation
                if cnt_opt % self.check_freq == 0 and cnt_opt > 0:
                    self.set_eval_mode()

            # Evaluate
            else:
                num_epi_run, _ = self.run_steps(num_epi=self.num_epi_per_eval)
                eval_reward_cum = self.eval_reward_cum_all / num_epi_run
                self.eval_record[self.cnt_step] = (eval_reward_cum, )

                # Report
                logging.info("======================================")
                logging.info(f'Evaluating at step {self.cnt_step}...')
                if verbose:
                    logging.info(f'Eps: {self.eps_schduler.get_variable()}')
                    logging.info(f'Eval - avg cumulative reward: {eval_reward_cum}')
                    logging.info(f'number of positive examples in the buffer: {len(torch.where(self.reward_buffer == 1)[0])}')
                if self.use_wandb:
                    wandb.log({
                        "AgentTrain - avg eval cumulative Reward": eval_reward_cum,
                    }, step=self.cnt_step, commit=True)

                # Saving modell and training details
                if self.save_metric == 'cum_reward':
                    best_path = self.save(metric=eval_reward_cum)
                else:
                    raise NotImplementedError
                torch.save(
                    {
                        'loss_record': self.loss_record,
                        'eval_record': self.eval_record,
                    }, os.path.join(self.out_folder, 'train_details'))

                # Generate sample affordance map
                for aff_ind in range(self.num_affordance):
                    img = self.obs_buffer[self.rng.integers(0, len(self.obs_buffer))].float()/255.0
                    img_pred = self.learner(img[None].to(self.device)).squeeze(1).squeeze(0)
                    save_affordance_map(img=img,
                                        pred=img_pred,
                                        path_prefix=os.path.join(self.img_folder, 
                                                                f'{self.cnt_step}_{aff_ind}'))

                # Switch to training
                self.set_train_mode()
                logging.info("======================================")

        ################### Done ###################
        best_path = self.save(force_save=True)
        return best_path


    # === Replay and update ===
    def sample_batch(self, batch_size=None, positive_ratio=None):
        # Sample indices
        if batch_size is None:
            batch_size = self.batch_size

        # Train by sampling from buffer
        buffer_size = self.obs_buffer.shape[0]
        if positive_ratio is None:
            sample_inds = random.sample(range(buffer_size), k=batch_size)
        else:
            positive_inds = torch.where(self.reward_buffer == 1)[0]
            negative_inds = torch.where(self.reward_buffer == 0)[0]
            num_positive = min(len(positive_inds), int(batch_size * positive_ratio))
            positive_sample_inds = positive_inds[random.sample(range(len(positive_inds)), 
                                                        k=min(num_positive, len(positive_inds)))]
            num_negative = batch_size - len(positive_sample_inds)
            negative_sample_inds = negative_inds[random.sample(range(len(negative_inds)), 
                                                               k=num_negative)]
            sample_inds = torch.cat((positive_sample_inds, negative_sample_inds))  
            sample_inds = sample_inds[torch.randperm(len(sample_inds))]
        
        obs_batch = self.obs_buffer[sample_inds].clone().detach().to(
            self.device, non_blocking=True)  # NxCxHxW
        ground_truth_batch = self.ground_truth_buffer[sample_inds].clone(
        ).detach().to(self.device, non_blocking=True)  # NxHxW
        mask_batch = self.mask_buffer[sample_inds].clone().detach().to(
            self.device, non_blocking=True)  # NxHxW
        return (obs_batch, ground_truth_batch, mask_batch)


    def unpack_batch(self, batch):
        return batch


    def store_transition(self, s, a, r, s_, done, info):
        """Save images as uint8"""

        # Indices to be replaced in the buffer for current step
        num_new, C, H, W = s.shape
        assert num_new == 1

        # Extract action
        _, _, _, theta, py, px, py_rot, px_rot = a
        reward_tensor = torch.tensor([r]).float()

        # Convert obs to tensor, uint8 to float32
        obs = s.detach().to('cpu')
        obs = obs.float()/255.0

        # Rotate to theta
        obs_rot = rotate_tensor(obs, theta=torch.tensor(theta))

        # Mark ground truth and mask with rotated x and y since depth is rotated (all zeros except for selected pixel)
        new_ground_truth = torch.zeros((1, H, W), dtype=torch.uint8).to('cpu')
        new_mask = torch.zeros((1, H, W), dtype=torch.uint8).to('cpu')
        new_ground_truth[0, int(py_rot), int(px_rot)] = int(r)
        new_mask[0, int(py_rot), int(px_rot)] = 1

        # Determine recency for new data
        # recency = np.exp(-self.cnt_step * 0.1)  # rank-based

        # # Debug
        # if r > 0:
        #     import matplotlib.pyplot as plt
        #     fig, axes = plt.subplots(1, 4)
        #     axes[0].imshow(obs[0,0].cpu().numpy())
        #     axes[0].scatter(px, py, c='r')
        #     axes[1].imshow(obs[0,1:].cpu().numpy().transpose(1,2,0))
        #     axes[2].imshow(obs_rot[0,0].cpu().numpy())
        #     axes[2].scatter(px_rot, py_rot, c='r')
        #     axes[3].imshow(obs_rot[0,1:].cpu().numpy().transpose(1,2,0))
        #     axes[0].set_title('Original depth (with action rotated back')
        #     axes[1].set_title('Original rgb')
        #     axes[2].set_title('Rotated depth')
        #     axes[3].set_title('Rotated rgb')
        #     plt.show()

        # Check if buffer filled up
        if self.obs_buffer.shape[0] < self.memory_capacity:
            self.obs_buffer = torch.cat(
                (self.obs_buffer, (obs_rot*255).byte()))[:self.memory_capacity]
            self.ground_truth_buffer = torch.cat(
                (self.ground_truth_buffer, new_ground_truth))[:self.memory_capacity]
            self.mask_buffer = torch.cat(
                (self.mask_buffer, new_mask))[:self.memory_capacity]
            # self.recency_buffer = np.concatenate(
            #     (self.recency_buffer, np.ones(
            #         (num_new)) * recency))[:self.memory_capacity]
            self.reward_buffer = torch.cat(
                (self.reward_buffer, reward_tensor))[:self.memory_capacity]
        else:
            # Replace older ones
            replace_ind = np.random.choice(self.memory_capacity,
                                           size=num_new,
                                           replace=False,
                                        #    p=self.recency_buffer /
                                        #    np.sum(self.recency_buffer)
                                           )
            self.obs_buffer[replace_ind] = (obs_rot*255).byte()
            self.ground_truth_buffer[replace_ind] = new_ground_truth
            self.mask_buffer[replace_ind] = new_mask
            # self.recency_buffer[replace_ind] = recency
            self.reward_buffer[replace_ind] = reward_tensor


    #== Reset policy/optimizer/memory
    def reset_policy(self, policy_path=None):
        if policy_path:
            self.learner.load_network(policy_path)
            logging.info('Loaded policy network from: {}'.format(policy_path))
        else:
            self.learner.build_network(self.cfg.learner.arch,   
                                       build_optimizer=False, 
                                       verbose=True)
            logging.info('Built new policy network!')


    def reset_memory(self, memory):
        if memory is not None:
            raise NotImplementedError
            # self.memory = memory
            # logging.info('Reusing memory with size {}!'.format(len(self.memory)))
        elif hasattr(self, 'memory_path'):
            raise NotImplementedError
        else:
            self.obs_buffer = torch.empty(
                (0, self.num_obs_channel, self.img_h, self.img_w), dtype=torch.uint8).to('cpu')
            self.ground_truth_buffer = torch.empty(
                (0, self.img_h, self.img_w), dtype=torch.uint8).to('cpu')
            self.mask_buffer = torch.empty(
                (0, self.img_h, self.img_w), dtype=torch.uint8).to('cpu')
            # self.recency_buffer = np.empty((0))
            self.reward_buffer = torch.empty((0)).float().to('cpu')
            logging.info('Built memory!')


    def reset_optimizer(self, optimizer_state=None):
        if optimizer_state:
            self.learner.load_optimizer_state(optimizer_state)
            logging.info('Loaded policy optimizer!')
        else:
            self.learner.build_optimizer()
            logging.info('Built new policy optimizer!')
