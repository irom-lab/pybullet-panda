import os
import numpy as np
import random
import torch
import logging
import wandb
import matplotlib.pyplot as plt

from agent.agent_base import AgentBase
from agent.learner import get_learner
from agent.utility import get_utility
from util.misc import load_obj
from util.image import save_affordance_map


class AgentImitate(AgentBase):
    """
    Run imitation learning using collected data. No more collecting experiences from environments.
    """
    def __init__(self, cfg, venv, verbose=True):
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
        self.max_opt = cfg.max_opt
        self.batch_size = cfg.batch_size
        self.num_update = cfg.num_update
        self.check_freq = cfg.check_freq
        self.num_epi_per_eval = cfg.num_eval_episode
        self.num_affordance = cfg.num_affordance
        self.expert_data_path = cfg.expert_data_path
        self.mask_grad_leakage = cfg.learner.mask_grad_leakage


    @property
    def cnt_step(self):
        return self.cnt_opt


    def learn(self, tasks=None, 
                    memory=None,
                    policy_path=None, 
                    optimizer_state=None,
                    verbose=False,
                    **kwargs):
        self.reset_save_info(self.out_folder)

        # Reset tasks
        if tasks is not None:
            self.reset_tasks(tasks, verbose)

        # Set up memory
        self.load_data(self.expert_data_path)

        # Set up policy
        self.reset_policy(policy_path)

        # Set up optimizer
        self.reset_optimizer(optimizer_state)

        # Run rest of steps while optimizing policy
        self.cnt_opt = 0
        while self.cnt_opt <= self.max_opt:
            print(self.cnt_opt, end='\r')
            self.set_train_mode()

            # Update policy 
            loss = 0
            for _ in range(self.num_update):
                loss_batch = self.learner.update(self.sample_batch())
                loss += loss_batch
            loss /= self.num_update
            self.loss_record[self.cnt_opt] = {'CE loss': loss}

            # Evaluate
            self.set_eval_mode()
            num_epi_run, _ = self.run_steps(num_epi=self.num_epi_per_eval, 
                                            force_deterministic=True)
            eval_reward_cum_avg = self.eval_reward_cum_all / num_epi_run
            self.eval_record[self.cnt_opt] = (eval_reward_cum_avg, )
            
            # Report
            logging.info("======================================")
            logging.info(f'Evaluating at step {self.cnt_opt}...')
            logging.info(f'Avg cumulative reward: {eval_reward_cum_avg}')
            if self.use_wandb:
                wandb.log({
                    "AgentImitate - CE loss": loss,
                    "AgentImitate - avg eval cumulative Reward": eval_reward_cum_avg,
                }, step=self.cnt_opt, commit=True)

            # Saving model and training detailss
            if self.save_metric == 'cum_reward':
                best_path = self.save(metric=eval_reward_cum_avg)
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
                                                             f'{self.cnt_opt}_{aff_ind}'))
            logging.info("======================================")

            # Count number of optimization
            self.cnt_opt += 1

        ################### Done ###################
        best_path = self.save(force_save=True)
        return best_path


    # === Replay and update ===
    def sample_batch(self, batch_size=None):
        # Sample indices
        if batch_size is None:
            batch_size = self.batch_size
        buffer_size = self.obs_buffer.shape[0]
        sample_inds = random.sample(range(buffer_size), k=batch_size)

        # Get data        
        obs_batch = self.obs_buffer[sample_inds].clone().detach().to(
            self.device, non_blocking=True)  # NxCxHxW
        ground_truth_batch = self.ground_truth_buffer[sample_inds].clone(
        ).detach().to(self.device, non_blocking=True)  # NxHxW
        mask_batch = self.mask_buffer[sample_inds].clone().detach().to(
            self.device, non_blocking=True)  # NxHxW
        return (obs_batch, ground_truth_batch, mask_batch)


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


    def load_data(self, path):
        """Imitation data. Obs in uint8. Assume all reward=1"""
        data = load_obj(path)
        self.obs_buffer = data['obs'].to('cpu')
        N, C, H, W = self.obs_buffer.shape
        action_all = data['action']
        
        # Construnct ground truth and mask (all zeros except for selected pixel)
        self.ground_truth_buffer = torch.zeros((N, H, W), dtype=torch.uint8).to('cpu')
        self.mask_buffer = torch.ones((N, H, W), dtype=torch.float32).to('cpu')*self.mask_grad_leakage
        for trial_ind, (py, px) in enumerate(action_all):
            py = int(py)
            px = int(px)
            self.ground_truth_buffer[trial_ind, py, px] = 1
            self.mask_buffer[trial_ind, py, px] = 1

            # if trial_ind < 20:
            #     depth = self.obs_buffer[trial_ind, 0].clone().numpy()
            #     plt.imshow(depth, origin="upper")
            #     plt.scatter(px, py) # use imshow origin
            #     plt.savefig(f'{trial_ind}_{py}_{px}.png')
            #     plt.close()

        self.reward_buffer = torch.ones((len(self.obs_buffer))).float().to('cpu')
        logging.info(f'Loaded imitation data from {path}!')


    def reset_optimizer(self, optimizer_state=None):
        if optimizer_state:
            self.learner.load_optimizer_state(optimizer_state)
            logging.info('Loaded policy optimizer!')
        else:
            self.learner.build_optimizer()
            logging.info('Built new policy optimizer!')
