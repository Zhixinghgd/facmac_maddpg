import logging
import os
import pickle
from copy import deepcopy
import numpy as np
import torch
import torch.nn.functional as F

from Agent import Agent
from Buffer import Buffer
from Facmac_agent import QMixNet
import logging


def setup_logger(filename):
    """
    Set up logger with filename.
    输入为log文件路径，输出为一条日志记录，使用示例：
        logger = setup_logger('logfile.log')  # 配置日志记录器，日志将保存到 'logfile.log'
        logger.info("This is an info message.")  # 记录一条 INFO 消息，对应文件中2025-01-02 10:00:00--INFO--This is an info message.
        logger.warning("This is a warning message.")  # 记录一条 WARNING 消息，2025-01-02 10:00:01--WARNING--This is a warning message.
    """
    logger = logging.getLogger()  # 获取一个全局日志记录器实例
    logger.setLevel(logging.INFO)  # 设置日志记录器的最低日志级别为 INFO,只有级别大于等于 INFO 的日志消息才会被记录。

    # 创建文件处理器，指定日志文件和写入模式
    handler = logging.FileHandler(filename, mode='a')  # 使用追加模式
    handler.setLevel(logging.INFO)  # 设置文件处理器的最低日志级别为 INFO

    # 设置日志格式：年月日时分秒--日志级别--日志内容
    formatter = logging.Formatter('%(asctime)s--%(levelname)s--%(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    handler.setFormatter(formatter)

    # 为了强制每次写入都立即刷新，覆盖原始的 `emit` 方法
    original_emit = handler.emit

    def flush_emit(record):
        original_emit(record)  # 执行原始的 emit 方法
        handler.flush()  # 强制刷新缓存

    handler.emit = flush_emit  # 用新的 emit 方法替换原始的 emit 方法

    # 将处理器添加到 logger 中
    logger.addHandler(handler)
    return logger


class MADDPG:
    """
    A MADDPG(Multi Agent Deep Deterministic Policy Gradient) agent
    Args:
        dim_info(dict):字典{agent_id：[obs_dim观测维度, act_dim动作维度]}.
        capacity(int):每个智能体经验回放缓冲区的最大容量,
        batch_size(int):每次采样的批量大小,
        actor_lr(float):actor 网络的学习率,
        critic_lr(float):critic 网络的学习率。
        res_dir(str):保存运行结果的文件夹路径
        num_good(int):逃跑者的数量
        num_adversaries(int):追逐者的数量
    """

    def __init__(self, dim_info, capacity, batch_size, actor_lr, critic_lr, res_dir, num_good, num_adversaries):
        # sum all the dims of each agent to get input dim for critic
        global_obs_act_dim = sum(sum(val) for val in dim_info.values())  # 所有智能体的观测维度和动作维度的和
        # dim_info.values()：返回字典中所有值，例如 [[obs_dim_0, act_dim_0], [obs_dim_1, act_dim_1]]
        # for val in dim_info.values()：遍历这些值，每个val是一个列表，如[obs_dim_0, act_dim_0]。
        # sum(val)：对每个val求和，例如obs_dim_0 + act_dim_0。sum(sum(val) for val in dim_info.values())：对所有val求和，例如obs_dim_0 + act_dim_0 + obs_dim_1 + act_dim_1。
        global_state_dim = 0
        # for val in dim_info.values():
        #     obs_dim, act_dim = val
        #     global_state_dim += obs_dim  # 假设obs_dim包含位置和速度的维度
        for agent_id, (obs_dim, act_dim) in dim_info.items():
            if agent_id.startswith("adversary_") or agent_id.startswith("leadadversary_"):
                global_state_dim += obs_dim
        # create Agent(actor-critic) and replay buffer for each agent
        self.agents = {}
        self.buffers = {}
        self.num_good = num_good
        self.num_adversaries = num_adversaries
        # for后面接多个参数是元组解包写法，但字典是以键值对的形式存储数据的，直接遍历字典时默认只返回键，所以用了items()方法，返回字典的键值对列表，然后解包为agent_id和(obs_dim, act_dim)。
        for agent_id, (obs_dim, act_dim) in dim_info.items():
            self.agents[agent_id] = Agent(agent_id, obs_dim, act_dim, global_obs_act_dim, actor_lr, critic_lr)
            self.buffers[agent_id] = Buffer(capacity, obs_dim, act_dim, 'cpu')
        self.total_reward_buffer = np.zeros(capacity, dtype=np.float32)
        self._index = 0  # 当前缓存写入的位置
        self._size = 0  # 当前缓存中的经验数量
        self.capacity = capacity

        self.dim_info = dim_info
        self.Mixing_net = QMixNet(
            state_shape=global_state_dim,  # 所有追逐者观测维度的和，不要动作维度
            hyper_hidden_dim=64,
            n_agents=num_adversaries,
            qmix_hidden_dim=32
        )
        self.Mixing_target_net = deepcopy(self.Mixing_net)

        # 定义优化器
        self.mixer_optimizer = torch.optim.Adam(
            self.Mixing_net.parameters(),
            lr=critic_lr  #应用qmix_lr， 暂用critic_lr代替
        )

        # 标识adversary
        # self.adversary_ids = [id for id in self.agents.keys()
        #                       if id.startswith("adversary_")]

        self.batch_size = batch_size
        self.res_dir = res_dir  # directory to save the training result
        self.logger = setup_logger(os.path.join(res_dir, 'maddpg.log'))

    def add(self, last_action, obs, action, reward, next_obs, done):  # obs, action, reward, next_obs, done都是字典，键为agent_id
        # NOTE that the experience is a dict with agent name as its key将一组经验（S、A、R、S1）添加到各自智能体的经验回放缓冲区中
        for agent_id in obs.keys():
            o = obs[agent_id]
            a = action[agent_id]
            l_a = last_action[agent_id]
            if isinstance(a, int):
                # the action from env.action_space.sample() is int, we have to convert it to onehot
                a = np.eye(self.dim_info[agent_id][1])[a]
                l_a = np.eye(self.dim_info[agent_id][1])[l_a]

            r = reward[agent_id]
            next_o = next_obs[agent_id]
            d = done[agent_id]
            self.buffers[agent_id].add(l_a, o, a, r, next_o, d)

    def add_total_reward(self, total_reward):
        self.total_reward_buffer[self._index] = total_reward
        self._index = (self._index + 1) % self.capacity
        if self._size < self.capacity:
            self._size += 1


    def sample(self, batch_size):
        """sample experience from all the agents' buffers, and collect data for network input"""
        # get the total num of transitions, these buffers should have same number of transitions用于采样经验
        total_num = len(self.buffers['agent_0'])  # 每个智能体的经验回放缓冲区中的经验数量应该相同，所以只计算第一个智能体的经验数量即可
        # randomly sample indices from all the transitions
        indices = np.random.choice(total_num, size=batch_size, replace=False)  # 从所有经验中随机采样batch_size个索引

        total_reward = self.total_reward_buffer[indices]
        total_reward = torch.from_numpy(total_reward).float().to('cpu')
        # NOTE that in MADDPG, we need the obs and actions of all agents
        # but only the reward and done of the current agent is needed in the calculation
        last_action, obs, act, reward, next_obs, done, next_act = {}, {}, {}, {}, {}, {}, {}
        for agent_id, buffer in self.buffers.items():
            l_a, o, a, r, n_o, d = buffer.sample(indices)  # 每个agent的buffer都采样相同的索引，indices是一个数组，包含batch_size个随机索引
            last_action[agent_id] = l_a
            obs[agent_id] = o #是被采样的智能体agent_id的所有被采样状态
            act[agent_id] = a
            reward[agent_id] = r
            next_obs[agent_id] = n_o
            done[agent_id] = d
            # calculate next_action using target_network and next_state
            next_act[agent_id] = self.agents[agent_id].target_action(n_o)
            # 经验池中采样次数编号是索引，存S、A、R、S、done。每个agent一个经验池，同一轮的所有智能体的经验在各自buffer中编号相同
            # 采样后的数据结构是以agent_id为键，值是采样后的数据，如obs[agent_id] = o，o是agent_id的所有被采样状态的数组
        return last_action, obs, act, reward, total_reward, next_obs, done, next_act

    def select_action(self, obs):
        actions = {}
        for agent, o in obs.items():
            o = torch.from_numpy(o).unsqueeze(0).float()  # 将观察值转换为tensor，并增加一个批次维度
            a = self.agents[agent].action(o)  # 这里的action是一个连续的实数值
            actions[agent] = a.squeeze(0).cpu().detach().numpy()  # 去掉批次维度并转换为numpy数组
            self.logger.info(f'{agent} action: {actions[agent]}')
        return actions

    def learn(self, batch_size, gamma, alpha=0.7):
        # 统一抽样
        last_act, obs, act, reward, total_reward, next_obs, done, next_act = self.sample(batch_size)

        def get_global(obs_dict):
            adversary_obs = [obs for id, obs in obs_dict.items() if id.startswith("adversary_")]
            return torch.cat(adversary_obs, dim=1)

        global_state = get_global(obs)
        next_global_state = get_global(next_obs)
        # MADDPG部分更新
        for agent_id, agent in self.agents.items():
            # 更新critic（保持原MADDPG逻辑）
            critic_value = agent.critic_value(list(obs.values()), list(act.values()))
            next_target = agent.target_critic_value(list(next_obs.values()), list(next_act.values()))
            target_value = reward[agent_id] + gamma * next_target * (1 - done[agent_id])
            critic_loss = F.mse_loss(critic_value, target_value.detach())
            agent.update_critic(critic_loss)

        # QMIX部分更新 需要限定只更新adversaries
        current_qs = torch.stack([
            agent.agent_q_value(obs[id], act[id])
            for id, agent in self.agents.items()if (id.startswith("adversary_") or id.startswith("leadadversary_"))], dim=1)  # (batch_size, n_adversaries)

        target_qs = torch.stack([
            agent.target_agent_q_value(next_obs[id], next_act[id])
            for id, agent in self.agents.items()if (id.startswith("adversary_") or id.startswith("leadadversary_"))], dim=1)


        q_tot = self.Mixing_net(current_qs, global_state)  # 混合网络输出
        target_q_tot = self.Mixing_target_net(target_qs, next_global_state)

        # 假设使用全局奖励（需与环境设置一致）
        # qmix_target = total_reward + gamma * target_q_tot * (1 - total_reward)
        # 上面需要(1 - done[agent_id])是防止已经结束的智能体的奖励干扰，全局只有一个应该不干扰吧？
        global_done = torch.stack(list(done.values())).any(dim=0).float()  #任一agent终止即视为全局终止
        qmix_target = total_reward + gamma * target_q_tot * (1 - global_done)
        qmix_loss = F.mse_loss(q_tot, qmix_target.detach())
        self.update_mixing(qmix_loss)  # 更新混合网络和局部Q 函数还没写，不知该放在MADDPG还是Facmac_agent

        # Actor更新（融合双Q）
        for agent_id, agent in self.agents.items():
            # 生成新动作
            new_action, logits = agent.action(obs[agent_id], model_out=True)
            new_act = {**act, agent_id: new_action}  # 创建新动作字典

            # 计算全局Q值
            q_global = agent.critic_value(list(obs.values()), list(new_act.values()))

            # 计算QMIX局部Q值
            if agent_id.startswith("adversary_") or agent_id.startswith("leadadversary_"):
                q_local = agent.agent_q_value(obs[agent_id], new_action)
                # 加权融合
                combined_q = alpha * q_global + (1 - alpha) * q_local
            else:
                combined_q = q_global

            actor_loss = -combined_q.mean() + 1e-3 * torch.pow(logits, 2).mean()

            agent.update_actor(actor_loss)


    def maddpg_learn(self, batch_size, gamma):
        for agent_id, agent in self.agents.items():
            last_act, obs, act, reward, total_reward, next_obs, done, next_act = self.sample(batch_size)
            # update critic
            critic_value = agent.critic_value(list(obs.values()), list(act.values()))

            # calculate target critic value
            next_target_critic_value = agent.target_critic_value(list(next_obs.values()),
                                                                 list(next_act.values()))
            target_value = reward[agent_id] + gamma * next_target_critic_value * (1 - done[agent_id])

            critic_loss = F.mse_loss(critic_value, target_value.detach(), reduction='mean')
            agent.update_critic(critic_loss)

            # update actor
            # action of the current agent is calculated using its actor
            action, logits = agent.action(obs[agent_id], model_out=True)
            act[agent_id] = action
            actor_loss = -agent.critic_value(list(obs.values()), list(act.values())).mean()  # 加上qmix的损失
            actor_loss_pse = torch.pow(logits, 2).mean()
            agent.update_actor(actor_loss + 1e-3 * actor_loss_pse)
            # self.logger.info(f'agent{i}: critic loss: {critic_loss.item()}, actor loss: {actor_loss.item()}')

    def qmix_learn(self, batch_size, gamma, alpha=0.7):
        # 统一抽样
        last_act, obs, act, reward, total_reward, next_obs, done, next_act = self.sample(batch_size)

        # def get_global(obs_dict):
        #     return torch.cat(list(obs_dict.values()), dim=1)

        def get_global(obs_dict):
            adversary_obs = [obs for id, obs in obs_dict.items() if id.startswith("adversary_")]
            return torch.cat(adversary_obs, dim=1)

        global_state = get_global(obs)
        next_global_state = get_global(next_obs)
        # MADDPG部分更新(仅agent)
        for agent_id, agent in self.agents.items():
            if agent_id.startswith("agent_") :
            # 更新critic（保持原MADDPG逻辑）
                critic_value = agent.critic_value(list(obs.values()), list(act.values()))
                next_target = agent.target_critic_value(list(next_obs.values()), list(next_act.values()))
                target_value = reward[agent_id] + gamma * next_target * (1 - done[agent_id])
                critic_loss = F.mse_loss(critic_value, target_value.detach())
                agent.update_critic(critic_loss)

        # QMIX部分更新 需要限定只更新adversaries
        current_qs = torch.stack([
            agent.agent_q_value(obs[id], act[id])
            for id, agent in self.agents.items()if (id.startswith("adversary") or id.startswith("leadadversary_"))], dim=1)  # (batch_size, n_adversaries)

        target_qs = torch.stack([
            agent.target_agent_q_value(next_obs[id], next_act[id])
            for id, agent in self.agents.items()if (id.startswith("adversary") or id.startswith("leadadversary_"))], dim=1)


        q_tot = self.Mixing_net(current_qs, global_state)  # 混合网络输出
        target_q_tot = self.Mixing_target_net(target_qs, next_global_state)

        # 假设使用全局奖励（需与环境设置一致）
        # qmix_target = total_reward + gamma * target_q_tot * (1 - total_reward)
        # 上面需要(1 - done[agent_id])是防止已经结束的智能体的奖励干扰，全局只有一个应该不干扰吧？
        global_done = torch.stack(list(done.values())).any(dim=0).float()  #任一agent终止即视为全局终止
        qmix_target = total_reward + gamma * target_q_tot * (1 - global_done)
        qmix_loss = F.mse_loss(q_tot, qmix_target.detach())
        self.update_mixing(qmix_loss)  # 更新混合网络和局部Q 函数还没写，不知该放在MADDPG还是Facmac_agent

        # Actor更新（融合双Q）
        for agent_id, agent in self.agents.items():
            # 生成新动作
            new_action, logits = agent.action(obs[agent_id], model_out=True)
            new_act = {**act, agent_id: new_action}  # 创建新动作字典

            # 计算全局Q值
            q_global = agent.critic_value(list(obs.values()), list(new_act.values()))

            # 计算QMIX局部Q值
            if agent_id.startswith("adversary") or agent_id.startswith("leadadversary_"):
                q_local = agent.agent_q_value(obs[agent_id], new_action)
                combined_q = q_local
            else:
                combined_q = q_global

            actor_loss = -combined_q.mean() + 1e-3 * torch.pow(logits, 2).mean()

            agent.update_actor(actor_loss)
    def update_mixing(self, qmix_loss):
        self.mixer_optimizer.zero_grad()
        for agent_id, agent in self.agents.items():
            if agent_id.startswith("adversary") or agent_id.startswith("leadadversary_"):
                self.agents[agent_id].q_agent_optimizer.zero_grad()

        qmix_loss.backward()

        # 混合网络梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.Mixing_net.parameters(), 10.0)

        # 各局部Q网络梯度裁剪
        for agent_id, agent in self.agents.items():
            if agent_id.startswith("adversary") or agent_id.startswith("leadadversary_"):
                torch.nn.utils.clip_grad_norm_(
                   self.agents[agent_id].q_agent.parameters(),
                   5.0
                )

        self.mixer_optimizer.step()
        for agent_id, agent in self.agents.items():
            if agent_id.startswith("adversary") or agent_id.startswith("leadadversary_"):
                self.agents[agent_id].q_agent_optimizer.step()
    # def update_mixing(self, qmix_loss):
    #     # 混合网络优化器清零梯度
    #     self.mixer_optimizer.zero_grad()
    #
    #     # 各adversary的Q网络清零梯度
    #     for agent_id, agent in self.agents.items():
    #         if agent_id.startswith("adversary_"):
    #             self.agents[agent_id].q_agent_optimizer.zero_grad()
    #
    #     # 反向传播
    #     qmix_loss.backward()
    #
    #     # 更新参数
    #     self.mixer_optimizer.step()
    #     for agent_id, agent in self.agents.items():
    #         if agent_id.startswith("adversary_"):
    #             self.agents[agent_id].q_agent_optimizer.step()



    def update_target(self, tau):
        def soft_update(from_network, to_network):
            """ copy the parameters of `from_network` to `to_network` with a proportion of tau"""
            for from_p, to_p in zip(from_network.parameters(), to_network.parameters()):
                to_p.data.copy_(tau * from_p.data + (1.0 - tau) * to_p.data)

        soft_update(self.Mixing_net, self.Mixing_target_net)
        for agent_id, agent in self.agents.items():
            soft_update(agent.actor, agent.target_actor)
            soft_update(agent.critic, agent.target_critic)
            if agent_id.startswith("adversary_") or agent_id.startswith("leadadversary_"):
                soft_update(agent.q_agent, agent.target_q_agent)

    def save(self, reward, total_rewards):
        """save actor parameters of all agents and training reward to `res_dir`"""
        torch.save(
            {name: agent.actor.state_dict() for name, agent in self.agents.items()},  # actor parameter
            os.path.join(self.res_dir, 'model.pt')
        )
        with open(os.path.join(self.res_dir, 'rewards.pkl'), 'wb') as f:  # save training data
            pickle.dump({'rewards': reward,
                         'total_rewards': total_rewards}, f)

    @classmethod
    def load(cls, dim_info, file, num_good, num_adversaries):
        """init maddpg using the model saved in `file`"""
        instance = cls(dim_info, 0, 0, 0, 0, os.path.dirname(file), num_good, num_adversaries)
        data = torch.load(file)
        for agent_id, agent in instance.agents.items():
            agent.actor.load_state_dict(data[agent_id])
        return instance
