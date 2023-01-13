import os
import numpy as np
import random
import torch
import logging
import wandb
import matplotlib.pyplot as plt

from agent.agent_imitate import AgentImitate
from util.image import save_affordance_map
from util.misc import load_obj


class AgentImitateEq(AgentImitate):
    def __init__(self, cfg, venv, verbose=True):
        """
        Run imitation learning with equivariance training
        """
        super().__init__(cfg, venv, verbose)


    def store_cfg(self, cfg):
        super().store_cfg(cfg)
        self.target_reward = cfg.target_reward
        

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

        # flag for checking whether training the action decoder or training the latent policy plus the state encoder - initially we train the action decoder
        flag_train_action_decoder = True 

        # Run rest of steps while optimizing policy
        self.cnt_opt = 0
        while self.cnt_opt <= self.max_opt:
            print(self.cnt_opt, end='\r')
            self.set_train_mode()

            # Update policy
            loss = np.empty((0,3))
            for update_ind in range(self.num_update):
                batch_train = self.sample_batch()
                if flag_train_action_decoder:
                    action_decoder_ce_loss = self.learner.update_action_decoder(batch_train, verbose=update_ind < 1)
                else:
                    action_decoder_ce_loss = 0
                latent_policy_ce_loss, alignment_loss, stats = self.learner.update_latent_policy(batch_train, verbose=update_ind < 1)
                loss = np.vstack((loss, [action_decoder_ce_loss, latent_policy_ce_loss, alignment_loss]))
            loss = np.sum(loss, axis=1) / self.num_update
            self.loss_record[self.cnt_opt] = {'action_decoder_ce_loss': loss[0],
                                               'latent_policy_ce_loss': loss[1],
                                               'alignment_loss': loss[2]}

            # Evaluate
            self.set_eval_mode()
            num_epi_run, _ = self.run_steps(num_epi=self.num_epi_per_eval, 
                                            force_deterministic=True)
            eval_reward_cum_avg = self.eval_reward_cum_all / num_epi_run
            self.eval_record[self.cnt_opt] = (eval_reward_cum_avg, )

            #! Check if target reward is reached
            flag_train_action_decoder = eval_reward_cum_avg < self.target_reward

            # Report
            logging.info("======================================")
            logging.info(f'Evaluating at step {self.cnt_opt}...')
            logging.info(f'Avg cumulative reward: {eval_reward_cum_avg}')
            logging.info(f'Training action decoder next? {flag_train_action_decoder}')
            if self.use_wandb:
                wandb.log({
                    "Imitation - action decoder ce loss": loss[0],
                    "Imitation - latent policy ce loss": loss[1],
                    "Imitation - alignment loss": loss[2],
                    "Eval - avg cumulative Reward": eval_reward_cum_avg,
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
                img = self.depth_buffer[self.rng.integers(0, self.depth_buffer.shape[0])].float()/255.0
                img_pred = self.learner(img[None, None].to(self.device)).squeeze(1).squeeze(0)
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
        buffer_size = self.depth_buffer.shape[0]
        sample_inds = random.sample(range(buffer_size), k=batch_size)

        # Get data        
        depth_batch = self.depth_buffer[sample_inds].clone().detach().to(
            self.device, non_blocking=True).unsqueeze(1)  # Nx1xHxW
        ground_truth_batch = self.ground_truth_buffer[sample_inds].clone(
        ).detach().to(self.device, non_blocking=True)  # NxHxW
        mask_batch = self.mask_buffer[sample_inds].clone().detach().to(
            self.device, non_blocking=True)  # NxHxW

        # Get action
        action_batch = self.action_buffer[sample_inds].clone().detach().to(
            self.device, non_blocking=True)  # Nx2
        return (depth_batch, ground_truth_batch, mask_batch, action_batch, None)    # None for reward batch


    def load_data(self, path):
        """Also save action as buffer. Not normalized."""
        super().load_data(path)
        
        # Load action
        data = load_obj(path)
        self.action_buffer = data['action'].float().to('cpu')
