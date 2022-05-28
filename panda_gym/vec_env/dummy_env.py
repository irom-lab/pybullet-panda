import torch


class DummyEnvPanda():
    def __init__(self, env, device):
        self.env = env
        self.device = device


    @property
    def state_dim(self):
        return self.env.state_dim


    @property
    def action_dim(self):
        return self.env.action_dim


    def reset(self, tasks):
        obs = self.env.reset(tasks[0])
        obs = torch.from_numpy(obs).unsqueeze(0).to(self.device)
        return obs


    def reset_one(self, env_ind, task):
        return self.reset([task])


    def step(self, action):
        action = action.cpu().numpy()
        obs, reward, done, info = self.env.step(action[0])
        obs = torch.from_numpy(obs).unsqueeze(0).to(self.device)
        reward = torch.tensor([reward]).unsqueeze(0).to(self.device)
        return obs, reward, [done], [info]


    def env_method(self, name):
        method = getattr(self.env, name)
        return method()
