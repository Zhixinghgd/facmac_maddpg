from copy import deepcopy
from typing import List

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.optim import Adam


class Agent:
    """Agent that can interact with environment from pettingzoo"""

    def __init__(self, agent_id, obs_dim, act_dim, global_obs_act_dim, actor_lr, critic_lr):
        self.actor = MLPNetwork(obs_dim, act_dim)
        # self.actor = MLPNetwork(obs_dim + act_dim, act_dim)
        # 后续actor考虑加入上一步动作MLPNetwork(obs_dim + act_dim（此处为上个动作，维度相同）, act_dim)

        # critic input all the observations and actions
        # if there are 3 agents for example, the input for critic is (obs1, obs2, obs3, act1, act2, act3)
        self.critic = MLPNetwork(global_obs_act_dim, 1)
        self.actor_optimizer = Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=critic_lr)
        self.target_actor = deepcopy(self.actor)
        self.target_critic = deepcopy(self.critic)
        self.agent_id = agent_id
        if agent_id.startswith("adversary_") or agent_id.startswith("leadadversary_"):  # 追逐者的Q部分添加QMIX思想
            print("This is an adversary.")
            self.q_agent = MLPNetwork(obs_dim + act_dim, 1)  # 输入应为单个agent的obs+action
            self.q_agent_optimizer = Adam(self.q_agent.parameters(), lr=critic_lr)
            self.target_q_agent = deepcopy(self.q_agent)
        elif agent_id.startswith("agent_"):  # 逃跑者由纯maddpg驱动
            print("This is an agent.")
        else:
            print("Unknown type.")

    @staticmethod
    def gumbel_softmax(logits, tau=1.0, eps=1e-20):
        # NOTE that there is a function like this implemented in PyTorch(torch.nn.functional.gumbel_softmax),
        # but as mention in the doc, it may be removed in the future, so i implement it myself
        epsilon = torch.rand_like(logits)
        logits += -torch.log(-torch.log(epsilon + eps) + eps)
        return F.softmax(logits / tau, dim=-1)

    def action(self, obs, model_out=False):
        """
        计算当前动作，针对连续动作空间，直接返回连续动作。
        """
        logits = self.actor(obs)  # logits 是 [batch_size, action_size] 的张量，表示连续动作

        # 使用 Sigmoid 将动作值限制在 [0.0, 1.0] 之间
        action = torch.sigmoid(logits)  # Sigmoid 确保动作在 [0, 1] 范围内

        if model_out:
            return action, logits
        return action

    def target_action(self, obs):
        """
        使用目标演员网络计算下一步动作。
        """
        logits = self.target_actor(obs)  # logits 是 [batch_size, action_size] 的张量，表示连续动作

        # 使用 Sigmoid 将动作值限制在 [0.0, 1.0] 之间
        action = torch.sigmoid(logits)  # Sigmoid 确保动作在 [0, 1] 范围内

        return action.squeeze(0).detach()  # 去掉批次维度，并返回动作

    def critic_value(self, state_list: List[Tensor], act_list: List[Tensor]):
        x = torch.cat(state_list + act_list, 1)
        return self.critic(x).squeeze(1)  # tensor with a given length

    def target_critic_value(self, state_list: List[Tensor], act_list: List[Tensor]):
        x = torch.cat(state_list + act_list, 1)
        return self.target_critic(x).squeeze(1)  # tensor with a given length

    def agent_q_value(self, state: Tensor, action: Tensor):  # 输入应为单个agent的state和action
        """为adversary设计的局部Q网络"""
        x = torch.cat([state, action], dim=1)
        return self.q_agent(x).squeeze(1)

    def target_agent_q_value(self, state: Tensor, action: Tensor):
        x = torch.cat([state, action], dim=1)
        return self.target_q_agent(x).squeeze(1)

    # def update_actor(self, loss):
    #     self.actor_optimizer.zero_grad()
    #     loss.backward()
    #     torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
    #     self.actor_optimizer.step()

    def update_actor(self, loss):  # 添加梯度控制以增强训练稳定性
        self.actor_optimizer.zero_grad()
        loss.backward()
        # 差异化梯度裁剪
        if self.agent_id.startswith("adversary_") or self.agent_id.startswith("leadadversary_"):
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.8)
        else:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_optimizer.step()

    def update_critic(self, loss):
        self.critic_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optimizer.step()
    #
    # def update_agent_q(self, loss):
    #     self.q_agent_optimizer.zero_grad()
    #     loss.backward()
    #     torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
    #     self.q_agent_optimizer.step()

class MLPNetwork(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=64, non_linear=nn.ReLU()):
        super(MLPNetwork, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            non_linear,
            nn.Linear(hidden_dim, hidden_dim),
            non_linear,
            nn.Linear(hidden_dim, out_dim),
        ).apply(self.init)

    @staticmethod
    def init(m):
        """init parameter of the module"""
        gain = nn.init.calculate_gain('relu')
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight, gain=gain)
            m.bias.data.fill_(0.01)

    def forward(self, x):
        return self.net(x)


