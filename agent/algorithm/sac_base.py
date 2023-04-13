import torch
import numpy as np

from abc import ABC, abstractmethod
import os
import copy
from torch.optim import Adam
from torch.optim import lr_scheduler

from network.sac import SACPiNetwork, SACTwinnedQNetwork
from util.scheduler import StepLRMargin
from util.network import soft_update, save_model


class SAC_Base(ABC):

    def __init__(self, CONFIG, CONFIG_ARCH, CONFIG_ENV):
        """
        __init__: initialization.

        Args:
            CONFIG (object): update-rekated hyper-parameter configuration.
            CONFIG_ARCH (object): NN architecture configuration.
            config_env (object): environment configuration.
        """
        self.CONFIG = CONFIG
        self.CONFIG_ARCH = CONFIG_ARCH
        self.EVAL = CONFIG.EVAL

        # == ENV PARAM ==
        self.action_mag = CONFIG_ENV.ACTION_MAG
        self.action_dim = CONFIG_ENV.ACTION_DIM
        self.img_h = CONFIG_ENV.CAMERA['img_h']
        self.img_w = CONFIG_ENV.CAMERA['img_w']

        # NN: device, action indicators
        self.device = CONFIG.DEVICE

        # reach-avoid setting
        self.mode = CONFIG.MODE

        # == PARAM FOR TRAINING ==
        if not self.EVAL:

            # Learning Rate
            self.LR_A_SCHEDULE = CONFIG.LR_A_SCHEDULE
            self.LR_C_SCHEDULE = CONFIG.LR_C_SCHEDULE
            if self.LR_A_SCHEDULE:
                self.LR_A_PERIOD = CONFIG.LR_A_PERIOD
                self.LR_A_DECAY = CONFIG.LR_A_DECAY
                self.LR_A_END = CONFIG.LR_A_END
            if self.LR_C_SCHEDULE:
                self.LR_C_PERIOD = CONFIG.LR_C_PERIOD
                self.LR_C_DECAY = CONFIG.LR_C_DECAY
                self.LR_C_END = CONFIG.LR_C_END
            self.LR_C = CONFIG.LR_C
            self.LR_A = CONFIG.LR_A

            # Discount factor
            self.GAMMA_SCHEDULE = CONFIG.GAMMA_SCHEDULE
            if self.GAMMA_SCHEDULE:
                self.GammaScheduler = StepLRMargin(
                    initValue=CONFIG.GAMMA, period=CONFIG.GAMMA_PERIOD,
                    decay=CONFIG.GAMMA_DECAY, endValue=CONFIG.GAMMA_END,
                    goalValue=1.
                )
                self.GAMMA = self.GammaScheduler.get_variable()
            else:
                self.GAMMA = CONFIG.GAMMA

            # Target Network Update
            self.TAU = CONFIG.TAU

            # alpha-related hyper-parameters
            self.init_alpha = CONFIG.ALPHA
            self.LEARN_ALPHA = CONFIG.LEARN_ALPHA
            self.log_alpha = torch.tensor(np.log(self.init_alpha)
                                         ).to(self.device)
            self.target_entropy = -self.action_dim
            if self.LEARN_ALPHA:
                self.log_alpha.requires_grad = True
                self.LR_Al = CONFIG.LR_Al
                self.LR_Al_SCHEDULE = CONFIG.LR_Al_SCHEDULE
                if self.LR_Al_SCHEDULE:
                    self.LR_Al_PERIOD = CONFIG.LR_Al_PERIOD
                    self.LR_Al_DECAY = CONFIG.LR_Al_DECAY
                    self.LR_Al_END = CONFIG.LR_Al_END
                print(
                    "SAC with learnable alpha and target entropy = {:.1e}".
                    format(self.target_entropy)
                )
            else:
                print("SAC with fixed alpha = {:.1e}".format(self.init_alpha))

    @property
    def alpha(self):
        return self.log_alpha.exp()

    @property
    @abstractmethod
    def has_latent(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def latent_dist(self):
        raise NotImplementedError

    def build_network(
        self, verbose=True, actor_path=None, critic_path=None, tie_conv=True
    ):
        self.actor = SACPiNetwork(
            input_n_channel=self.CONFIG_ARCH.OBS_CHANNEL,
            img_sz=[self.img_h, self.img_w],
            # latent_dim=self.CONFIG.LATENT_DIM,
            action_dim=self.action_dim,
            action_mag=self.action_mag,
            mlp_dim=self.CONFIG_ARCH.MLP_DIM['actor'],
            append_dim=self.CONFIG_ARCH.APPEND_DIM,
            activation_type=self.CONFIG_ARCH.ACTIVATION['actor'],
            kernel_sz=self.CONFIG_ARCH.KERNEL_SIZE,
            stride=self.CONFIG_ARCH.STRIDE,
            padding=self.CONFIG_ARCH.PADDING,
            n_channel=self.CONFIG_ARCH.N_CHANNEL,
            use_sm=self.CONFIG_ARCH.USE_SM,
            use_ln=self.CONFIG_ARCH.USE_LN,
            device=self.device,
            verbose=verbose
        )
        self.critic = SACTwinnedQNetwork(
            input_n_channel=self.CONFIG_ARCH.OBS_CHANNEL,
            img_sz=[self.img_h, self.img_w],
            # latent_dim=self.CONFIG.LATENT_DIM,
            mlp_dim=self.CONFIG_ARCH.MLP_DIM['critic'],
            action_dim=self.action_dim,
            append_dim=self.CONFIG_ARCH.APPEND_DIM,
            activation_type=self.CONFIG_ARCH.ACTIVATION['critic'],
            kernel_sz=self.CONFIG_ARCH.KERNEL_SIZE,
            stride=self.CONFIG_ARCH.STRIDE,
            padding=self.CONFIG_ARCH.PADDING,
            n_channel=self.CONFIG_ARCH.N_CHANNEL,
            use_sm=self.CONFIG_ARCH.USE_SM,
            use_ln=self.CONFIG_ARCH.USE_LN,
            device=self.device,
            verbose=verbose
        )

        # Load model if specified
        if actor_path is not None:
            self.actor.load_state_dict(
                torch.load(actor_path, map_location=self.device)
            )
            print("--> Load actor wights from {}".format(actor_path))
        if critic_path is not None:
            self.critic.load_state_dict(
                torch.load(critic_path, map_location=self.device)
            )
            print("--> Load critic wights from {}".format(critic_path))

        # Copy for critic targer
        self.critic_target = copy.deepcopy(self.critic)

        # Tie weights for conv layers
        if tie_conv:
            self.actor.encoder.copy_conv_weights_from(self.critic.encoder)

    def build_optimizer(self):
        print("Build basic optimizers.")
        self.critic_optimizer = Adam(self.critic.parameters(), lr=self.LR_C)
        self.actor_optimizer = Adam(self.actor.parameters(), lr=self.LR_A)

        if self.LR_C_SCHEDULE:
            self.critic_scheduler = lr_scheduler.StepLR(
                self.critic_optimizer, step_size=self.LR_C_PERIOD,
                gamma=self.LR_C_DECAY
            )
        if self.LR_A_SCHEDULE:
            self.actor_scheduler = lr_scheduler.StepLR(
                self.actor_optimizer, step_size=self.LR_A_PERIOD,
                gamma=self.LR_A_DECAY
            )

        if self.LEARN_ALPHA:
            self.log_alpha_optimizer = Adam([self.log_alpha], lr=self.LR_Al)
            if self.LR_Al_SCHEDULE:
                self.log_alpha_scheduler = lr_scheduler.StepLR(
                    self.log_alpha_optimizer, step_size=self.LR_Al_PERIOD,
                    gamma=self.LR_Al_DECAY
                )

    # region: update functions
    def update_alpha_hyperParam(self):
        if self.LR_Al_SCHEDULE:
            lr = self.log_alpha_optimizer.state_dict()['param_groups'][0]['lr']
            if lr <= self.LR_Al_END:
                for param_group in self.log_alpha_optimizer.param_groups:
                    param_group['lr'] = self.LR_Al_END
            else:
                self.log_alpha_scheduler.step()

    def update_critic_hyperParam(self):
        if self.LR_C_SCHEDULE:
            lr = self.critic_optimizer.state_dict()['param_groups'][0]['lr']
            if lr <= self.LR_C_END:
                for param_group in self.critic_optimizer.param_groups:
                    param_group['lr'] = self.LR_C_END
            else:
                self.critic_scheduler.step()
        if self.GAMMA_SCHEDULE:
            self.GammaScheduler.step()
            self.GAMMA = self.GammaScheduler.get_variable()

    def update_actor_hyperParam(self):
        if self.LR_A_SCHEDULE:
            lr = self.actor_optimizer.state_dict()['param_groups'][0]['lr']
            if lr <= self.LR_A_END:
                for param_group in self.actor_optimizer.param_groups:
                    param_group['lr'] = self.LR_A_END
            else:
                self.actor_scheduler.step()

    def update_hyper_param(self):
        self.update_critic_hyperParam()
        self.update_actor_hyperParam()
        if self.LEARN_ALPHA:
            self.update_alpha_hyperParam()

    def update_target_networks(self):
        soft_update(self.critic_target, self.critic, self.TAU)

    @abstractmethod
    def update_actor(self, batch):
        raise NotImplementedError

    @abstractmethod
    def update_critic(self, batch):
        raise NotImplementedError

    @abstractmethod
    def update(self, batch, timer, update_period=2):
        raise NotImplementedError

    # utils
    @abstractmethod
    def value(self, obs, append):
        raise NotImplementedError

    def save(self, step, logs_path, max_model=None):
        path_c = os.path.join(logs_path, 'critic')
        path_a = os.path.join(logs_path, 'actor')
        save_model(self.critic, step, path_c, 'critic', max_model)
        save_model(self.actor, step, path_a, 'actor', max_model)

    def remove(self, step, logs_path):
        path_c = os.path.join(
            logs_path, 'critic', 'critic-{}.pth'.format(step)
        )
        path_a = os.path.join(logs_path, 'actor', 'actor-{}.pth'.format(step))
        print("Remove", path_a)
        print("Remove", path_c)
        if os.path.exists(path_c):
            os.remove(path_c)
        if os.path.exists(path_a):
            os.remove(path_a)

    # def check(self, env, cnt_step, states, check_type, verbose=True, **kwargs):
    #     if self.mode == 'safety' or self.mode == 'risk':
    #         end_type = 'fail'
    #     elif self.mode == 'RA':
    #         end_type = 'safety_ra'
    #     else:
    #         end_type = 'TF'

    #     self.actor.eval()
    #     self.critic.eval()

    #     if check_type == 'random':
    #         results = env.simulate_trajectories(self,
    #                                             mode=self.mode,
    #                                             states=states,
    #                                             end_type=end_type,
    #                                             **kwargs)[1]
    #     elif check_type == 'all_env':
    #         results = env.simulate_all_envs(self,
    #                                         mode=self.mode,
    #                                         states=states,
    #                                         end_type=end_type,
    #                                         **kwargs)[1]
    #     else:
    #         raise ValueError(
    #             "Check type ({}) not supported!".format(check_type))
    #     if self.mode == 'safety' or self.mode == 'risk':
    #         failure = np.sum(results == -1) / results.shape[0]
    #         success = 1 - failure
    #         trainProgress = np.array([success, failure])
    #     else:
    #         success = np.sum(results == 1) / results.shape[0]
    #         failure = np.sum(results == -1) / results.shape[0]
    #         unfinish = np.sum(results == 0) / results.shape[0]
    #         trainProgress = np.array([success, failure, unfinish])

    #     if verbose:
    #         print('\n{} policy after [{}] steps:'.format(self.mode, cnt_step))
    #         if not self.EVAL:
    #             print('  - gamma={:.6f}, alpha={:.1e}.'.format(
    #                 self.GAMMA, self.alpha))
    #         if self.mode == 'safety' or self.mode == 'risk':
    #             print('  - success/failure ratio:', end=' ')
    #         else:
    #             print('  - success/failure/unfinished ratio:', end=' ')
    #         with np.printoptions(formatter={'float': '{: .2f}'.format}):
    #             print(trainProgress)
    #     self.actor.train()
    #     self.critic.train()

    #     return trainProgress

    # # endregion
