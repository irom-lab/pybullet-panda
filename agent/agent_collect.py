import os
import numpy as np
import torch
import logging

from agent.agent_train import AgentTrain
from util.image import rotate_tensor
from util.scheduler import StepLRFixed
from util.misc import save_obj


class AgentCollect(AgentTrain):
    def __init__(self, cfg, venv, verbose=True):
        """
        Collect expert data for imitation. Learner can be scripted, or (to be implemented).
        """
        super().__init__(cfg, venv)
        self.action_dim = 2

        # Not used
        self.eps_schduler = StepLRFixed(initValue=self.cfg_eps.init,
                                        period=1000,
                                        endValue=self.cfg_eps.end,
                                        stepSize=self.cfg_eps.step)


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

        # Run initial steps with random steps
        self.set_train_mode()
        num_batch = self.max_sample_steps // self.batch_size
        for batch_ind in range(num_batch):
            self.run_steps(self.batch_size, force_random=False)
            logging.info(f'Batch {batch_ind+1} out of {num_batch}, number of positive grasps in the buffer: {len(torch.where(self.reward_buffer == 1)[0])}')

        # Save buffer
        positive_ind = torch.where(self.reward_buffer == 1)[0]
        positive_depth = self.depth_buffer[positive_ind]
        positive_action = self.action_buffer[positive_ind]
        data = {'depth': positive_depth, 'action': positive_action}
        save_obj(data, os.path.join(self.out_folder, 'data.pkl'))
        logging.info(f'Saved {len(positive_ind)} positive grasps!')
        return None


    def store_transition(self, s, a, r, s_, done, info):
        """Save images as uint8"""

        # Indices to be replaced in the buffer for current step
        num_new, _, H, W = s.shape
        assert num_new == 1

        # Extract action - note that the rotated pixels are the ones actually executed
        x, y, z, theta, py, px, py_rot, px_rot = a
        action_tensor = torch.tensor([[py_rot, px_rot]]).float()
        reward_tensor = torch.tensor([r]).float()

        # Convert depth to tensor, uint8 to float32
        depth = s.detach().to('cpu')
        depth = depth.float()/255.0

        # Rotate to theta
        depth_rot = rotate_tensor(depth, theta=torch.tensor(theta)).squeeze(1)

        # # Debug
        # print('Executed pos (should match dot in the original view): ', (x,y,z))
        # import matplotlib.pyplot as plt
        # fig, axes = plt.subplots(1, 2)
        # axes[0].imshow(depth.squeeze(0).squeeze(0))
        # axes[0].scatter(px, py, c='r')
        # axes[1].imshow(depth_rot.squeeze(0))
        # axes[1].scatter(px_rot, py_rot, c='r')
        # axes[0].set_title('Original')
        # axes[1].set_title('Rotated (saved)')
        # plt.show()

        # append, assume not filled up
        self.depth_buffer = torch.cat(
            (self.depth_buffer, (depth_rot*255).byte()))[:self.memory_capacity]
        self.action_buffer = torch.cat(
            (self.action_buffer, action_tensor))[:self.memory_capacity]
        self.reward_buffer = torch.cat(
            (self.reward_buffer, reward_tensor))[:self.memory_capacity]


    def reset_memory(self, memory):
        """Also building action buffer"""
        super().reset_memory(memory)

        self.action_buffer = torch.empty((0, self.action_dim)).float().to('cpu')
        logging.info('Built action buffer!')
