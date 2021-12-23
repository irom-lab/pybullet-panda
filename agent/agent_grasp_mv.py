# Please contact the author(s) of this library if you have any questions.
# Authors: Kai-Chieh Hsu ( kaichieh@princeton.edu )
#          Allen Z. Ren (allen.ren@princeton.edu)

import os
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb

from alano.train.sac_mini import SAC_mini
from alano.train.agent_base import AgentBase


class AgentGraspMV(AgentBase):
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
        self.policy = SAC_mini(CONFIG_UPDATE, CONFIG_ARCH, CONFIG_ENV)
        self.policy.build_network(verbose=verbose)

        # alias
        self.module_all = [self.policy]
        self.performance = self.policy

    def learn(self, venv, current_step=None):

        # hyper-parameters
        max_steps = self.CONFIG.MAX_STEPS
        opt_freq = self.CONFIG.OPTIMIZE_FREQ
        num_update_per_opt = self.CONFIG.UPDATE_PER_OPT
        check_opt_freq = self.CONFIG.CHECK_OPT_FREQ
        min_step_b4_opt = self.CONFIG.MIN_STEPS_B4_OPT
        out_folder = self.CONFIG.OUT_FOLDER
        # num_rnd_traj = self.CONFIG.NUM_VALIDATION_TRAJS

        # == Main Training ==
        start_learning = time.time()
        train_records = []
        train_progress = [[], []]
        violation_record = []
        episode_record = []
        cnt_opt = 0
        cnt_opt_period = 0
        cnt_safety_violation = 0
        cnt_num_episode = 0

        # Saving model
        model_folder = os.path.join(out_folder, 'model')
        os.makedirs(model_folder, exist_ok=True)
        self.module_folder_all = [model_folder]
        save_metric = self.CONFIG.SAVE_METRIC

        # Figure folder
        # figure_folder = os.path.join(out_folder, 'figure')
        # os.makedirs(figure_folder, exist_ok=True)

        if current_step is None:
            self.cnt_step = 0
        else:
            self.cnt_step = current_step
            print("starting from {:d} steps".format(self.cnt_step))

        # Reset all envs
        s, _ = venv.reset()

        # Steps
        while self.cnt_step <= max_steps:
            print(self.cnt_step, end='\r')

            # Set train modes for all envs
            venv.env_method('set_train_mode')

            # Get append
            append_all = venv.get_append(venv.get_attr('_state'))

            # Select action
            with torch.no_grad():
                a_all, _ = self.policy.actor.sample(s,
                                                    append=append_all,
                                                    latent=None)

            # Apply action - update heading
            s_all, r_all, done_all, info_all = venv.step(a_all)

            # Get new append
            append_nxt_all = venv.get_append(venv.get_attr('_state'))

            # Check all envs
            for env_ind, (s_, r, done, info) in enumerate(
                    zip(s_all, r_all, done_all, info_all)):

                # Save append
                info['append'] = append_all[env_ind].unsqueeze(0)
                info['append_nxt'] = append_nxt_all[env_ind].unsqueeze(0)

                # Store the transition in memory
                action = a_all[env_ind].unsqueeze(0).clone()
                self.store_transition(
                    s[env_ind].unsqueeze(0).to(self.image_device), action, r,
                    s_.unsqueeze(0).to(self.image_device), done, info)

                # Check finished env
                if done:
                    obs, _ = venv.reset_one(env_ind, verbose=False)
                    s_all[env_ind] = obs
                    cnt_num_episode += 1
            episode_record.append(cnt_num_episode)

            # Update "prev" states
            s = s_all

            # Optimize
            if (self.cnt_step >= min_step_b4_opt
                    and cnt_opt_period >= opt_freq):
                cnt_opt_period = 0

                # Update critic/actor
                loss = np.zeros(4)

                for timer in range(num_update_per_opt):
                    batch = self.unpack_batch(self.sample_batch(),
                                              get_latent=False)

                    loss_tp = self.policy.update(
                        batch, timer, update_period=self.UPDATE_PERIOD)
                    for i, l in enumerate(loss_tp):
                        loss[i] += l
                loss /= num_update_per_opt

                # Record: loss_q, loss_pi, loss_entropy, loss_alpha
                train_records.append(loss)
                if self.CONFIG.USE_WANDB:
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
                # venv.env_method('close_pb')
                # s, _ = venv.reset(random_init=random_init)

                # Count number of optimization
                cnt_opt += 1

                # Check after fixed number of steps
                if cnt_opt % check_opt_freq == 0:
                    print()

                    # Release GPU RAM as much as possible
                    torch.cuda.empty_cache()

                    # Set states for check()
                    # sample_states = self.get_check_states(env, num_rnd_traj)
                    # progress = self.policy.check(
                    #     venv,
                    #     self.cnt_step,
                    #     states=sample_states,
                    #     check_type='random',
                    #     verbose=True,
                    #     revert_task=True,
                    #     sample_task=True,
                    #     num_rnd_traj=num_rnd_traj,
                    # )
                    progress = [0, 0]
                    train_progress[0].append(progress)
                    train_progress[1].append(self.cnt_step)
                    if self.CONFIG.USE_WANDB:
                        wandb.log(
                            {
                                "Success": progress[0],
                                "cnt_num_episode": cnt_num_episode,
                            },
                            step=self.cnt_step,
                            commit=True)

                    # Saving model
                    if save_metric == 'perf':
                        self.save(metric=progress[0])
                    else:
                        raise NotImplementedError

                    # Save training details
                    torch.save(
                        {
                            'train_records': train_records,
                            'train_progress': train_progress,
                            "episode_record": episode_record,
                        }, os.path.join(out_folder, 'train_details'))

                    # Release GPU RAM as much as possible
                    torch.cuda.empty_cache()

                    # # Re-initialize env
                    # env.close_pb()
                    # env.reset()

            # Count
            self.cnt_step += self.n_envs
            cnt_opt_period += self.n_envs

            # Update gamma, lr etc.
            for _ in range(self.n_envs):
                self.policy.updateHyperParam()

        self.save(force_save=True)
        end_learning = time.time()
        time_learning = end_learning - start_learning
        print('\nLearning: {:.1f}'.format(time_learning))

        train_records = np.array(train_records)
        for i, tp in enumerate(train_progress):
            train_progress[i] = np.array(tp)
        episode_record = np.array(episode_record)
        return train_records, train_progress, violation_record, episode_record
