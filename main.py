import argparse
import os
import gym
import matplotlib.pyplot as plt
import numpy as np

from pettingzoo.mpe import simple_adversary_v2, simple_spread_v2, simple_tag_v2, simple_world_comm_v2

from MADDPG import MADDPG


def get_env(env_name, ep_len=25):
    """create environment and get observation and action dimension of each agent in this environment"""
    new_env = None
    num_good = 2
    num_adversaries = 5
    if env_name == 'simple_adversary_v2':
        new_env = simple_adversary_v2.parallel_env(max_cycles=ep_len)
    if env_name == 'simple_spread_v2':
        new_env = simple_spread_v2.parallel_env(max_cycles=ep_len)
    if env_name == 'simple_tag_v2':
        new_env = simple_tag_v2.parallel_env(num_good=num_good, num_adversaries=num_adversaries, num_obstacles=4, max_cycles=ep_len, continuous_actions=True)
        # new_env = simple_tag_v2.parallel_env(max_cycles=ep_len)
    if env_name == 'simple_world_comm_v2':
        new_env = simple_world_comm_v2.parallel_env(num_good=num_good, num_adversaries=num_adversaries, num_obstacles=1,
                num_food=2, max_cycles=25, num_forests=2, continuous_actions=True)

    new_env.reset()  # 初始化环境
    _dim_info = {}


    # 获取 agent 的动作空间
    action_space = new_env.action_space('agent_0')
    # 判断动作空间类型
    if isinstance(action_space, gym.spaces.Discrete):
        # print(f"{'agent_0'} 的动作空间是离散的，动作数量为 {action_space.n}")
        for agent_id in new_env.agents:
            _dim_info[agent_id] = []  # [obs_dim, act_dim]第agent_id号智能体的观察信息维度、动作信息维度
            _dim_info[agent_id].append(new_env.observation_space(agent_id).shape[0])  # .append()在列表末尾添加元素
            _dim_info[agent_id].append(new_env.action_space(agent_id).n)
    elif isinstance(action_space, gym.spaces.Box):
        # print(f"{'agent_0'} 的动作空间是连续的，动作维度为 {action_space.shape}")
        for agent_id in new_env.agents:
            _dim_info[agent_id] = []  # [obs_dim, act_dim]第agent_id号智能体的观察信息维度、动作信息维度
            _dim_info[agent_id].append(new_env.observation_space(agent_id).shape[0])  # .append()在列表末尾添加元素
            _dim_info[agent_id].append(new_env.action_space(agent_id).shape[0])

    #     原本的
    # for agent_id in new_env.agents:
    #     _dim_info[agent_id] = []  # [obs_dim, act_dim]第agent_id号智能体的观察信息维度、动作信息维度
    #     _dim_info[agent_id].append(new_env.observation_space(agent_id).shape[0])#.append()在列表末尾添加元素
    #     _dim_info[agent_id].append(new_env.action_space(agent_id).n)

        # if isinstance(new_env.action_space(agent_id), int):  # 创建环境参数continuous_actions=False，动作空间为离散的
        #     _dim_info[agent_id].append(new_env.action_space(agent_id).n)
        # else:  # 创建环境参数continuous_actions=True，动作空间为连续的   ！！！判断逻辑有问题，会把离散动作空间也判断为连续动作空间
        #     _dim_info[agent_id].append(new_env.action_space(agent_id).shape[0])

    return new_env, _dim_info, num_good, num_adversaries


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 解析命令行参数
    parser.add_argument('env_name', type=str, default='simple_adversary_v2', help='name of the env',
                        choices=['simple_adversary_v2', 'simple_spread_v2', 'simple_tag_v2', 'simple_world_comm_v2'])
    parser.add_argument('--episode_num', type=int, default=30000,
                        help='total episode num during training procedure')
    parser.add_argument('--episode_length', type=int, default=25, help='steps per episode')
    parser.add_argument('--learn_interval', type=int, default=100,
                        help='steps interval between learning time')
    parser.add_argument('--random_steps', type=int, default=1e4,
                        help='random steps before the agent start to learn')
    parser.add_argument('--tau', type=float, default=0.02, help='soft update parameter')
    parser.add_argument('--gamma', type=float, default=0.95, help='discount factor')
    parser.add_argument('--buffer_capacity', type=int, default=int(1e6), help='capacity of replay buffer')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch-size of replay buffer')
    parser.add_argument('--actor_lr', type=float, default=0.01, help='learning rate of actor')
    parser.add_argument('--critic_lr', type=float, default=0.01, help='learning rate of critic')
    args = parser.parse_args()

    # create folder to save result
    env_dir = os.path.join('./results', args.env_name)
    if not os.path.exists(env_dir):
        os.makedirs(env_dir)
    total_files = len([file for file in os.listdir(env_dir)])
    result_dir = os.path.join(env_dir, f'{total_files + 1}')
    os.makedirs(result_dir)

    env, dim_info, num_good, num_adversaries= get_env(args.env_name, args.episode_length)
    maddpg = MADDPG(dim_info, args.buffer_capacity, args.batch_size, args.actor_lr, args.critic_lr,
                    result_dir, num_good, num_adversaries)

    step = 0  # global step counter
    agent_num = env.num_agents
    # reward of each episode of each agent 创建一个字典，键为agent_id，值为一个长度为args.episode_num的数组，np.zeros()初始化为全0，存每轮奖励
    episode_rewards = {agent_id: np.zeros(args.episode_num) for agent_id in env.agents}
    #episode_rewards['agent_1']   输出：array([0.0, 0.0, ..., 0.0])，长度为 30000
    episode_total_rewards = np.zeros(args.episode_num)
    for episode in range(args.episode_num):
        obs = env.reset()  # obs是一个字典，键为agent_id，值为该智能体的初始状态
        agent_reward = {agent_id: 0 for agent_id in env.agents}  # agent reward of the current episode
        last_action = {agent_id: None for agent_id in env.agents}  # last action of the current episode
        r_total_reward = 0
        while env.agents:  # interact with the env for an episode ，env.agents是一个列表，每次 env.step(action) 后，环境会更新 env.agents，移除完成的智能体。当所有智能体完成（env.agents 为空），循环结束，进入下一个回合。
            step += 1
            if step < args.random_steps:#随机探索阶段：如果总步数小于 args.random_steps，为每个智能体随机采样动作。每个episode开始先随机走出args.random_steps步
                action = {agent_id: env.action_space(agent_id).sample() for agent_id in env.agents}
            else:
                # print(f'use maddpg')未知原因导致1900 episode后才开始使用maddpg--原因是计数器step是全局计数跨episode不清零
                action = maddpg.select_action(obs)#智能策略阶段：超过随机探索步数后，使用 MADDPG 模型根据观察状态选择动作。

            next_obs, reward, total_reward, done, info = env.step(action)#并行执行Parallel
            # total_reward = info["agent_0"]["total_reward"]  # 从任意agent的info中获取
            # print(f"Individual Rewards: {reward}, Total Reward: {total_reward}")
            # env.render()
            maddpg.add(last_action, obs, action, reward, next_obs, done)#将当前步骤的数据存储到经验回放池中，为后续学习提供样本。
            maddpg.add_total_reward(total_reward)
            #obs：当前状态S，action：当前动作A，reward：当前奖励R，next_obs：下一个状态S'，done：各个智能体是否完成
            for agent_id, r in reward.items():  # update reward，将每个智能体的当前步骤奖励累加到其总奖励。
                agent_reward[agent_id] += r

            if step >= args.random_steps and step % args.learn_interval == 0:  # learn every few steps同一训练步进行的，可以考虑分开
                maddpg.learn(args.batch_size, args.gamma)#更新评估网络Q和策略网络A
                # maddpg.qmix_learn(args.batch_size, args.gamma)
                # maddpg.maddpg_learn(args.batch_size, args.gamma)
                maddpg.update_target(args.tau)#目标网络更新：使用软更新参数 args.tau 更新目标网络的权重。

            r_total_reward += total_reward
            obs = next_obs
            last_action = action

        # episode finishes
        for agent_id, r in agent_reward.items():  # record reward
            episode_rewards[agent_id][episode] = r

        episode_total_rewards[episode] = r_total_reward

        if (episode + 1) % 100 == 0:  # print info every 100 episodes
            message = f'episode {episode + 1}, '
            sum_reward = 0
            for agent_id, r in agent_reward.items():  # record reward
                message += f'{agent_id}: {r:>4f}; '
                sum_reward += r
            message += f'sum reward: {sum_reward}'
            print(message)

    maddpg.save(episode_rewards,episode_total_rewards)  # save model


    def get_running_reward(arr: np.ndarray, window=100):
        """
        计算滑动平均奖励。
        输入：
            arr (np.ndarray): 奖励数组
            window (int): 滑动窗口的大小（默认值为100）
        输出：
            running_reward (np.ndarray): 滑动平均奖励数组
        """
        # 初始化一个与输入数组形状相同的零数组，用于存储滑动平均奖励
        running_reward = np.zeros_like(arr)

        # 对前 window-1 个元素，计算它们的平均值（不足 window 的长度，取从头到当前元素的平均值）
        for i in range(window - 1):
            running_reward[i] = np.mean(arr[:i + 1])

        # 对剩余的元素，使用滑动窗口计算平均值
        for i in range(window - 1, len(arr)):
            running_reward[i] = np.mean(arr[i - window + 1:i + 1])

        return running_reward


    # 训练完成后，绘制奖励曲线
    fig, ax = plt.subplots()  # 创建图表和轴
    x = range(1, args.episode_num + 1)  # x 轴是训练的 episode 数，从 1 到 episode_num

    # 遍历每个智能体的奖励
    for agent_id, reward in episode_rewards.items():
        ax.plot(x, reward, label=agent_id)  # 绘制原始奖励曲线
        ax.plot(x, get_running_reward(reward))  # 绘制滑动平均奖励曲线

    ax.legend()  # 添加图例，用于区分不同智能体的奖励曲线
    ax.set_xlabel('episode')  # x 轴标签
    ax.set_ylabel('reward')  # y 轴标签
    title = f'training result of maddpg solve {args.env_name}'  # 图表标题
    ax.set_title(title)  # 设置图表标题

    # 保存图表到 result_dir 目录下
    plt.savefig(os.path.join(result_dir, title))  # 保存图表

