import os
from queue import PriorityQueue
import logging
import numpy as np
import pickle5 as pickle
import torch


class AgentBase():
    def __init__(self, cfg, venv):

        # Params
        self.venv = venv
        self.cfg = cfg
        self.seed = cfg.seed
        self.rng = np.random.default_rng(seed=cfg.seed)
        self.device = cfg.device
        self.image_device = cfg.image_device
        self.eval = cfg.eval

        # Params for vectorized envs
        self.n_envs = cfg.num_cpus
        self.action_dim = cfg.action_dim
        self.max_train_steps = cfg.max_train_steps
        self.max_eval_steps = cfg.max_eval_steps
        self.env_step_cnt = [0 for _ in range(self.n_envs)]

        # Params for saving
        self.save_top_k = cfg.save_top_k
        self.save_metric = cfg.save_metric
        self.use_wandb = cfg.use_wandb
        self.out_folder = cfg.out_folder
        self.reset_save_info(self.out_folder)
        self.flag_save_memory = cfg.save_memory
        self.flag_save_optim = cfg.save_optim

        # Load tasks
        if hasattr(cfg, 'dataset'):
            logging.info("= loading tasks from {}".format(cfg.dataset))
            with open(cfg.dataset, 'rb') as f:
                task_all = pickle.load(f)
            self.reset_tasks(task_all)
        else:
            logging.warning('= No dataset loaded!')


    def reset_save_info(self, out_folder):
        os.makedirs(out_folder, exist_ok=True)
        self.module_folder_all = [out_folder]
        self.pq_top_k = PriorityQueue()

        # Save memory
        self.memory_folder = os.path.join(out_folder, 'memory')
        os.makedirs(self.memory_folder, exist_ok=True)
        self.optim_folder = os.path.join(out_folder, 'optim')
        os.makedirs(self.optim_folder, exist_ok=True)

        # Save loss and eval info, key is step number
        self.loss_record = {}
        self.eval_record = {}


    def reset_tasks(self, tasks, verbose=True):
        self.task_all = tasks
        self.num_task = len(self.task_all)
        if verbose:
            logging.info(f"{self.num_task} tasks are loaded")


    def set_train_mode(self):
        self.eval_mode = False
        self.max_env_step = self.max_train_steps
        self.learner.eval = False


    def set_eval_mode(self):
        self.eval_reward_cumulative = [0 for _ in range(self.n_envs)
                                       ]  # for calculating cumulative reward
        self.eval_reward_best = [0 for _ in range(self.n_envs)]
        self.eval_reward_cumulative_all = 0
        self.eval_reward_best_all = 0
        self.env_step_cnt = [0 for _ in range(self.n_envs)]

        self.eval_mode = True
        self.max_env_step = self.max_eval_steps
        self.learner.eval = True


    def run_steps(self, num_step=None, 
                        num_episode=None, 
                        force_random=False, 
                        run_in_seq=False):
        if num_step is not None:
            cnt_target = num_step
        elif num_episode is not None:
            cnt_target = num_episode
        else:
            raise "No steps or episodes provided for run_steps()!"

        if run_in_seq:
            task_ids_yet_run = list(np.arange(num_episode))

        # Run
        info_episode = []
        cnt = 0
        while cnt < cnt_target:

            # Reset
            if run_in_seq:
                new_ids = task_ids_yet_run[:self.n_envs]
                task_ids_yet_run = task_ids_yet_run[self.n_envs:]
                s, task_ids = self.reset_env_all(new_ids)
            else:
                s, task_ids = self.reset_env_all()

            # Interact
            # cur_tasks = [self.task_all[id] for id in task_ids]
            # if self.use_append:
            #     append_all = self.utility.get_append(cur_tasks)
            # else:
            #     append_all = None

            # Select action
            eps = self.eps_schduler.get_variable()
            if force_random:
                flag_random = 1
            else:
                flag_random = self.rng.choice(2, p=[1-eps, eps])
            with torch.no_grad():
                a_all = self.learner.forward(s,
                                            #  append=append_all,
                                            flag_random=flag_random,
                                            )

            # Apply action - update heading
            s_all, r_all, done_all, info_all = self.step(a_all)

            # # Get new append
            # append_nxt_all = None

            # Check all envs
            for env_ind, (s_, r, done, info) in enumerate(
                    zip(s_all, r_all, done_all, info_all)):

                # # Save append
                # if append_all is not None:
                #     info['append'] = append_all[env_ind].unsqueeze(0)
                # if append_nxt_all is not None:
                #     info['append_nxt'] = append_nxt_all[env_ind].unsqueeze(0)

                # Store the transition in memory if training mode - do not save next state
                action = a_all[env_ind]
                if not self.eval_mode:
                    self.store_transition(
                        s[env_ind].unsqueeze(0).to(self.image_device), 
                        action, r, None, done, info)

                # Increment step count for the env
                self.env_step_cnt[env_ind] += 1

                # Check reward
                if self.eval_mode:
                    self.eval_reward_cumulative[env_ind] += r.item()
                    self.eval_reward_best[env_ind] = max(self.eval_reward_best[env_ind], r.item())
                    
                    # Check done for particular env
                    if done or self.env_step_cnt[env_ind] > self.max_env_step:
                        info['reward'] = self.eval_reward_cumulative[env_ind]
                        self.eval_reward_cumulative_all += self.eval_reward_cumulative[env_ind]
                        self.eval_reward_best_all += self.eval_reward_best[env_ind]
                        self.eval_reward_cumulative[env_ind] = 0
                        self.eval_reward_best[env_ind] = 0

                        # Record info of the episode
                        info_episode += [info]

                        # Count for eval mode
                        cnt += 1
                        
                        # Quit
                        if cnt == cnt_target:
                            return cnt, info_episode
                    else:
                        if done or self.env_step_cnt[env_ind] > self.max_env_step:
                            info['reward'] = r.item()   # assume single step!

                            # Record info of the episode
                            info_episode += [info]

            # Count for train mode
            if not self.eval_mode:
                cnt += self.n_envs

                # Update gamma, lr etc.
                for _ in range(self.n_envs):
                    self.learner.update_hyper_param()
                    self.eps_schduler.step()
        return cnt, info_episode


    # === Venv ===
    def step(self, action):
        return self.venv.step(action)


    def reset_sim(self):
        self.venv.env_method('close_pb')


    def reset_env_all(self, task_ids=None, verbose=False):
        if task_ids is None:
            task_ids = self.rng.integers(low=0,
                                         high=self.num_task,
                                         size=(self.n_envs, ))

        # fill if not enough
        if len(task_ids) < self.n_envs:
            num_yet_fill = self.n_envs - len(task_ids)
            task_ids += [0 for _ in range(num_yet_fill)]

        tasks = [self.task_all[id] for id in task_ids]
        s = self.venv.reset(tasks)
        if verbose:
            for index in range(self.n_envs):
                logging.info("<-- Reset environment {} with task {}:".format(
                    index, task_ids[index]))
        self.env_step_cnt = [0 for _ in range(self.n_envs)]
        return s, task_ids


    def reset_env(self, env_ind, task_id=None, verbose=False):
        if task_id is None:
            task_id = self.rng.integers(low=0, high=self.num_task)
        task = self.task_all[task_id]
        s = self.venv.reset_one(index=env_ind, task=task)
        if verbose:
            logging.info("<-- Reset environment {} with task {}:".format(
                env_ind, task))
        self.env_step_cnt[env_ind] = 0
        return s, task_id


    # === Models ===
    def save(self, metric=0, force_save=False):
        assert metric is not None or force_save, "should provide metric of force save"
        save_current = True
        if force_save or self.pq_top_k.qsize() < self.save_top_k:
            self.pq_top_k.put((metric, self.cnt_step))
        elif metric > self.pq_top_k.queue[0][0]:  # overwrite entry with lowest metric (index=0)
            # Remove old one
            _, step_remove = self.pq_top_k.get()
            for module, module_folder in zip(self.module_all,
                                             self.module_folder_all):
                module.remove(int(step_remove), module_folder)

            # Remove memory
            if self.flag_save_memory:
                path_memory = os.path.join(self.memory_folder, 'memory-{}.pt'.format(int(step_remove)))
                if os.path.exists(path_memory):
                    os.remove(path_memory)
            if self.flag_save_optim:
                path_optim = os.path.join(self.optim_folder, 'optim-{}.pt'.format(int(step_remove)))
                if os.path.exists(path_optim):
                    os.remove(path_optim)
            self.pq_top_k.put((metric, self.cnt_step))
        else:
            save_current = False

        if save_current:
            for module, module_folder in zip(self.module_all,
                                             self.module_folder_all):
                path = module.save(self.cnt_step, module_folder)

            # Save replay buffer
            if self.flag_save_memory:
                path_memory = os.path.join(self.memory_folder, 'memory-{}.pt'.format(self.cnt_step))
                self.memory.save(path_memory)
            if self.flag_save_optim:
                path_optim = os.path.join(self.optim_folder, 'optim-{}.pt'.format(self.cnt_step))
                self.learner.save_optimizer_state(path_optim)

        # always return the best path!  # todo minor: fix hack
        return os.path.join(self.module_folder_all[0], 'critic', 'critic-{}.pth'.format(self.pq_top_k.queue[-1][1])) 


    def save_memory(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.memory.memory, f, pickle.HIGHEST_PROTOCOL)


    # TODO: right now assumes critic
    def restore(self, step, logs_path, agent_type):
        """Restore the weights of the neural network.

        Args:
            step (int): #updates trained.
            logs_path (str): the path of the directory, under this folder there
                should be critic/ and agent/ folders.
        """
        model_folder = path_c = os.path.join(logs_path, agent_type)
        path_c = os.path.join(model_folder, 'critic',
                              'critic-{}.pth'.format(step))
        self.learner.critic.load_state_dict(
            torch.load(path_c, map_location=self.device))
        logging.info('  <= Restore {} with {} updates from {}.'.format(
            agent_type, step, model_folder))
