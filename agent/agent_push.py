import os
import time
import numpy as np
import torch
import wandb

from alano.algorithm.sac_mini import SAC_mini
from alano.train.agent_base import AgentBase


class AgentPush(AgentBase):
    def __init__(self,
                 CONFIG,
                 CONFIG_UPDATE,
                 CONFIG_ARCH,
                 CONFIG_ENV,
                 verbose=True):
        """
        __init__: initialization.

        Args:
            CONFIG (Class object): hyper-parameter configuration.
            verbose (bool, optional): print info or not. Defaults to True.
        """
        super().__init__(CONFIG, CONFIG_ENV, CONFIG_UPDATE)

        print("= Constructing policy agent")
        obs_channel = 0
        if CONFIG_ENV.USE_RGB:
            obs_channel += 3
        if CONFIG_ENV.USE_DEPTH:
            obs_channel += 1
        CONFIG_ARCH.OBS_CHANNEL = obs_channel
        self.policy = SAC_mini(CONFIG_UPDATE, CONFIG_ARCH, CONFIG_ENV)
        self.policy.build_network(verbose=verbose, 
                                build_optimizer=not self.eval,
                                actor_path=CONFIG_UPDATE.ACTOR_PATH, 
                                critic_path=CONFIG_UPDATE.CRITIC_PATH)

        # alias
        self.module_all = [self.policy]
        self.performance = self.policy

    def learn(self, venv, current_step=None):
        start_learning = time.time()
        train_records = []
        train_progress = [[], []]
        cnt_opt = 0
        cnt_opt_period = -(self.min_step_b4_opt - self.opt_freq)
        num_train_episode = 0
        success_threshold = 0.9

        # Saving model
        model_folder = os.path.join(self.out_folder, 'model')
        os.makedirs(model_folder, exist_ok=True)
        self.module_folder_all = [model_folder]

        # Figure folder
        # figure_folder = os.path.join(out_folder, 'figure')
        # os.makedirs(figure_folder, exist_ok=True)

        if current_step is None:
            self.cnt_step = 0
        else:
            self.cnt_step = current_step
            print("starting from {:d} steps".format(self.cnt_step))

        # Reset all envs
        self.venv = venv
        if self.eval:
            s = self.set_eval_mode()
        else:
            s = self.set_train_mode()

        # Steps
        while self.cnt_step <= self.max_sample_steps:
            print(self.cnt_step, end='\r')

            ################### Quit eval mode ###################
            if self.num_eval_episode >= self.num_episode_per_eval:
                eval_success = self.num_eval_success / self.num_eval_episode
                eval_reward_cumulative = self.eval_reward_cumulative_all / self.num_eval_episode
                eval_reward_best = self.eval_reward_best_all / self.num_eval_episode
                print('success rate: ', eval_success)
                print('avg cumulative reward: ', eval_reward_cumulative)
                print('avg best reward: ', eval_reward_best)
                train_progress[0].append([eval_success, eval_reward_cumulative, eval_reward_best])
                train_progress[1].append(self.cnt_step)
                if self.use_wandb:
                    wandb.log({
                        "Success": eval_success,
                        "Cumulative Reward": eval_reward_cumulative,
                        "Best Reward": eval_reward_best,
                    }, step=self.cnt_step, commit=True)

                # Saving model
                if self.save_metric == 'success':
                    self.save(metric=eval_success)
                elif self.save_metric == 'cum_reward':
                    self.save(metric=eval_reward_cumulative)
                elif self.save_metric == 'best_reward':
                    self.save(metric=eval_reward_best)
                else:
                    raise NotImplementedError

                # Save training details
                torch.save(
                    {
                        'train_records': train_records,
                        'train_progress': train_progress,
                    }, os.path.join(self.out_folder, 'train_details'))

                # Switch to training
                s = self.set_train_mode()

            ################### Interact ###################
            append_all = None

            # Select action
            with torch.no_grad():                
                a_all = self.forward(s, append=append_all, latent=None)

            # Apply action - update heading
            s_all, r_all, done_all, info_all = venv.step(a_all)

            # Get new append
            append_nxt_all = None

            # Check all envs
            for env_ind, (s_, r, done, info) in enumerate(
                    zip(s_all, r_all, done_all, info_all)):

                # Save append
                if append_all is not None:
                    info['append'] = append_all[env_ind].unsqueeze(0)
                    info['append_nxt'] = append_nxt_all[env_ind].unsqueeze(0)

                # Store the transition in memory
                action = a_all[env_ind].unsqueeze(0).clone()
                self.store_transition(
                    s[env_ind].unsqueeze(0).to(self.image_device), action, r,
                    s_.unsqueeze(0).to(self.image_device), done, info)

                # Increment step count for the env
                self.env_step_cnt[env_ind] += 1

                # Check reward
                if self.eval_mode:
                    self.eval_reward_cumulative[env_ind] += r.item()
                    self.eval_reward_best[env_ind] = max(self.eval_reward_best[env_ind], r.item())

                # Check done for particular env
                if done or self.env_step_cnt[env_ind] > self.max_env_step:
                    self.env_step_cnt[env_ind] = 0
                    if self.eval_mode:
                        self.num_eval_episode += 1
                        self.num_eval_success += (self.eval_reward_best[env_ind] > success_threshold)
                        self.eval_reward_cumulative_all += self.eval_reward_cumulative[env_ind]
                        self.eval_reward_best_all += self.eval_reward_best[env_ind]
                        self.eval_reward_cumulative[env_ind] = 0
                        self.eval_reward_best[env_ind] = 0
                    else:
                        num_train_episode += 1

                    # Reset
                    obs, _ = venv.reset_one(env_ind, verbose=False)
                    s_all[env_ind] = obs

            # Update "prev" states
            s = s_all

            ################### Optimize ###################
            if cnt_opt_period >= self.opt_freq:
                cnt_opt_period = 0

                # Update critic/actor
                loss = np.zeros(4)

                for timer in range(self.num_update_per_opt):
                    batch = self.unpack_batch(self.sample_batch())
                    loss_tp = self.policy.update(
                        batch, timer, update_period=self.update_period)
                    for i, l in enumerate(loss_tp):
                        loss[i] += l
                loss /= self.num_update_per_opt

                # Record: loss_q, loss_pi, loss_entropy, loss_alpha
                train_records.append(loss)
                if self.use_wandb:
                    wandb.log(
                        {
                            "loss_q": loss[0],
                            "loss_pi": loss[1],
                            "loss_entropy": loss[2],
                            "loss_alpha": loss[3],
                        },
                        step=self.cnt_step,
                        commit=False)

                # Re-initialize pb to avoid memory explosion from mesh loading
                # - this will also terminates any trajectories at sampling
                venv.env_method('close_pb')
                s, _ = venv.reset()

                # Count number of optimization
                cnt_opt += 1

                ################### Eval ###################
                if cnt_opt % self.check_opt_freq == 0:
                    s = self.set_eval_mode()

            # Count
            if not self.eval_mode:
                self.cnt_step += self.n_envs
                cnt_opt_period += self.n_envs

                # Update gamma, lr etc.
                for _ in range(self.n_envs):
                    self.policy.update_hyper_param()

        ################### Done ###################
        self.save(force_save=True)
        end_learning = time.time()
        time_learning = end_learning - start_learning
        print('\nLearning: {:.1f}'.format(time_learning))

        train_records = np.array(train_records)
        for i, tp in enumerate(train_progress):
            train_progress[i] = np.array(tp)
        return train_records, train_progress

    def evaluate(self, venv):
        return self.learn(venv)
