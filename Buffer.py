import numpy as np
import torch


class Buffer:
    """replay buffer for each agent"""

    def __init__(self, capacity, obs_dim, act_dim, device):
        self.capacity = capacity  # 缓存的最大容量

        self.last_action = np.zeros((capacity, act_dim))   # 上一个动作的数组，形状为(capacity, act_dim)

        self.obs = np.zeros((capacity, obs_dim))  # 存储观察值的数组，形状为 (capacity, obs_dim)
        self.action = np.zeros((capacity, act_dim))  # 存储动作的数组，形状为 (capacity, act_dim)
        self.reward = np.zeros(capacity)  # 存储奖励的数组，形状为 (capacity,)
        self.next_obs = np.zeros((capacity, obs_dim))  # 存储下一个观察值的数组，形状为 (capacity, obs_dim)
        self.done = np.zeros(capacity, dtype=bool)  # 存储done标志的数组，形状为 (capacity,)

        self._index = 0  # 当前缓存写入的位置
        self._size = 0  # 当前缓存中的经验数量

        self.device = device  # 存储设备，通常是 'cpu' 或 'cuda'

    def add(self,last_action, obs, action, reward, next_obs, done):
        """ add an experience to the memory """
        self.last_action[self._index] = last_action
        self.obs[self._index] = obs
        self.action[self._index] = action
        self.reward[self._index] = reward
        self.next_obs[self._index] = next_obs
        self.done[self._index] = done

        self._index = (self._index + 1) % self.capacity
        if self._size < self.capacity:
            self._size += 1

    def sample(self, indices):
        # retrieve data, Note that the data stored is ndarray
        last_action = self.last_action[indices]
        obs = self.obs[indices]
        action = self.action[indices]
        reward = self.reward[indices]
        next_obs = self.next_obs[indices]
        done = self.done[indices]

        # NOTE that `obs`, `action`, `next_obs` will be passed to network(nn.Module),
        # so the first dimension should be `batch_size`
        last_action = torch.from_numpy(last_action).float().to(self.device)  # torch.Size([batch_size, state_dim])
        obs = torch.from_numpy(obs).float().to(self.device)  # torch.Size([batch_size, state_dim])
        action = torch.from_numpy(action).float().to(self.device)  # torch.Size([batch_size, action_dim])
        reward = torch.from_numpy(reward).float().to(self.device)  # just a tensor with length: batch_size
        # reward = (reward - reward.mean()) / (reward.std() + 1e-7)
        next_obs = torch.from_numpy(next_obs).float().to(self.device)  # Size([batch_size, state_dim])
        done = torch.from_numpy(done).float().to(self.device)  # just a tensor with length: batch_size

        return last_action, obs, action, reward, next_obs, done

    def __len__(self):
        return self._size
