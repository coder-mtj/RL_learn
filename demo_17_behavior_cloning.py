# 导入gymnasium库，这是gym的维护版本，用于创建强化学习环境
import gymnasium as gym  # 使用gymnasium替代已弃用的gym
# 导入PyTorch深度学习框架
import torch
# 导入PyTorch的函数式接口，包含激活函数、损失函数等
import torch.nn.functional as F
# 导入PyTorch的神经网络模块
import torch.nn as nn
# 导入numpy库，用于数值计算和数组操作
import numpy as np
# 导入matplotlib绘图库，用于绘制训练曲线
import matplotlib.pyplot as plt
# 导入tqdm库，用于显示训练进度条
from tqdm import tqdm
# 导入随机数生成库，用于设置随机种子和随机采样
import random
# 导入自定义的强化学习工具模块
import rl_utils


# 定义策略网络类，继承自PyTorch的nn.Module
class PolicyNet(torch.nn.Module):
    """策略网络

    用于输出动作概率分布的神经网络。

    Args:
        state_dim (int): 状态空间的维度
        hidden_dim (int): 隐藏层的神经元数量
        action_dim (int): 动作空间的维度
    """

    def __init__(self, state_dim, hidden_dim, action_dim):
        """初始化策略网络的各个层

        Args:
            state_dim (int): 状态空间的维度
            hidden_dim (int): 隐藏层的神经元数量
            action_dim (int): 动作空间的维度
        """
        super(PolicyNet, self).__init__()  # 调用父类nn.Module的初始化方法
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)  # 创建第一个全连接层，输入维度为状态维度，输出维度为隐藏层维度
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)  # 创建第二个全连接层，输入维度为隐藏层维度，输出维度为动作维度

    def forward(self, x):
        """前向传播函数，计算给定状态下的动作概率分布

        Args:
            x (torch.Tensor): 输入的状态张量

        Returns:
            torch.Tensor: 动作概率分布
        """
        x = F.relu(self.fc1(x))  # 将输入x通过第一个全连接层fc1，然后应用ReLU激活函数
        return F.softmax(self.fc2(x), dim=1)  # 将x通过第二个全连接层fc2，然后应用softmax函数在维度1上得到概率分布


# 定义价值网络类，继承自PyTorch的nn.Module
class ValueNet(torch.nn.Module):
    """价值网络

    用于估计状态价值的神经网络。

    Args:
        state_dim (int): 状态空间的维度
        hidden_dim (int): 隐藏层的神经元数量
    """

    def __init__(self, state_dim, hidden_dim):
        """初始化价值网络的各个层

        Args:
            state_dim (int): 状态空间的维度
            hidden_dim (int): 隐藏层的神经元数量
        """
        super(ValueNet, self).__init__()  # 调用父类nn.Module的初始化方法
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)  # 创建第一个全连接层，输入维度为状态维度，输出维度为隐藏层维度
        self.fc2 = torch.nn.Linear(hidden_dim, 1)  # 创建第二个全连接层，输入维度为隐藏层维度，输出维度为1（单个价值）

    def forward(self, x):
        """前向传播函数，计算给定状态的价值

        Args:
            x (torch.Tensor): 输入的状态张量

        Returns:
            torch.Tensor: 状态价值
        """
        x = F.relu(self.fc1(x))  # 将输入x通过第一个全连接层fc1，然后应用ReLU激活函数
        return self.fc2(x)  # 将x通过第二个全连接层fc2，得到状态价值并返回


class PPO:
    ''' PPO算法,采用截断方式 '''
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                 lmbda, epochs, eps, gamma, device):
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)  # 创建策略网络并移动到指定设备
        self.critic = ValueNet(state_dim, hidden_dim).to(device)  # 创建价值网络并移动到指定设备
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)  # 创建策略网络的Adam优化器
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)  # 创建价值网络的Adam优化器
        self.gamma = gamma  # 保存折扣因子
        self.lmbda = lmbda  # 保存GAE参数lambda
        self.epochs = epochs  # 保存每次更新时的训练轮数
        self.eps = eps  # 保存PPO算法中的截断参数epsilon
        self.device = device  # 保存计算设备

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)  # 将状态转换为浮点型张量并移动到设备
        probs = self.actor(state)  # 通过策略网络获取动作概率分布
        action_dist = torch.distributions.Categorical(probs)  # 创建分类分布对象
        action = action_dist.sample()  # 从分布中采样一个动作
        return action.item()  # 将张量转换为Python标量并返回

    def update(self, transition_dict):
        # 将状态数据转换为tensor并移动到指定设备
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)
        # 将动作数据转换为tensor，指定数据类型为int64，调整形状为列向量并移动到指定设备
        actions = torch.tensor(transition_dict['actions'], dtype=torch.long).view(-1, 1).to(
            self.device)
        # 将奖励数据转换为tensor，调整形状为列向量并移动到指定设备
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        # 将下一状态数据转换为tensor并移动到指定设备
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)
        # 将结束标志转换为tensor，调整形状为列向量并移动到指定设备
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)
        # 计算TD目标值
        td_target = rewards + self.gamma * self.critic(next_states) * (1 -
                                                                       dones)
        # 计算TD误差
        td_delta = td_target - self.critic(states)
        # 计算优势函数
        advantage = rl_utils.compute_advantage(self.gamma, self.lmbda,
                                               td_delta.cpu()).to(self.device)
        # 计算旧策略的对数概率
        old_log_probs = torch.log(self.actor(states).gather(1,
                                                            actions)).detach()

        for _ in range(self.epochs):  # 循环进行指定轮数的训练
            log_probs = torch.log(self.actor(states).gather(1, actions))  # 计算当前策略的对数概率
            ratio = torch.exp(log_probs - old_log_probs)  # 计算重要性采样比率
            surr1 = ratio * advantage  # 计算第一个代理目标
            surr2 = torch.clamp(ratio, 1 - self.eps,
                                1 + self.eps) * advantage  # 计算截断后的第二个代理目标
            actor_loss = torch.mean(-torch.min(surr1, surr2))  # 计算PPO策略损失函数
            critic_loss = torch.mean(
                F.mse_loss(self.critic(states), td_target.detach()))  # 计算价值网络的均方误差损失
            self.actor_optimizer.zero_grad()  # 清零策略网络优化器的梯度
            self.critic_optimizer.zero_grad()  # 清零价值网络优化器的梯度
            actor_loss.backward()  # 反向传播计算策略网络的梯度
            critic_loss.backward()  # 反向传播计算价值网络的梯度
            self.actor_optimizer.step()  # 更新策略网络的参数
            self.critic_optimizer.step()  # 更新价值网络的参数


actor_lr = 1e-3  # 设置策略网络的学习率为0.001
critic_lr = 1e-2  # 设置价值网络的学习率为0.01
num_episodes = 500  # 设置训练回合数为50（减少用于快速测试）
hidden_dim = 128  # 设置神经网络隐藏层的维度为128
gamma = 0.98  # 设置折扣因子为0.98
lmbda = 0.95  # 设置GAE（广义优势估计）参数lambda为0.95
epochs = 10  # 设置每次更新时的训练轮数为10
eps = 0.2  # 设置PPO算法中的截断参数epsilon为0.2
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")  # 检查CUDA可用性，选择计算设备
print(f"使用设备: {device}")  # 打印当前使用的计算设备

env_name = 'CartPole-v1'  # 设置环境名称为CartPole-v1（使用v1版本替代已弃用的v0版本）
env = gym.make(env_name)  # 创建强化学习环境实例
env.reset(seed=0)  # 重置环境并设置随机种子为0（gymnasium的新语法）
torch.manual_seed(0)  # 设置PyTorch的随机种子为0以确保结果可复现
state_dim = env.observation_space.shape[0]  # 获取状态空间的维度
action_dim = env.action_space.n  # 获取动作空间的维度（离散动作数量）
ppo_agent = PPO(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda,
                epochs, eps, gamma, device)  # 创建PPO智能体实例

return_list = rl_utils.train_on_policy_agent(env, ppo_agent, num_episodes)  # 使用PPO算法训练智能体并获取每回合的回报列表



# 定义专家数据采样函数
def sample_expert_data(n_episode):
    """使用训练好的PPO智能体采样专家数据

    Args:
        n_episode (int): 采样的回合数

    Returns:
        tuple: (states, actions) 专家状态和动作数据
    """
    # 初始化状态和动作列表
    states = []
    actions = []
    # 循环采样指定回合数的数据
    for episode in range(n_episode):
        # 重置环境，获取初始状态
        state, _ = env.reset()  # gymnasium返回(observation, info)
        # 初始化结束标志
        done = False
        # 在一个回合内循环采样
        while not done:
            # 使用PPO智能体选择动作
            action = ppo_agent.take_action(state)
            # 保存当前状态和动作
            states.append(state)
            actions.append(action)
            # 执行动作，获取下一状态和奖励
            next_state, reward, done, truncated, _ = env.step(action)  # gymnasium返回5个值
            # 更新当前状态
            state = next_state
            # 检查是否因为时间限制而结束
            if truncated:
                done = True
    # 返回numpy数组格式的状态和动作数据
    return np.array(states), np.array(actions)


env.reset(seed=0)  # 重置环境并设置随机种子为0（gymnasium的新语法）
torch.manual_seed(0)  # 设置PyTorch的随机种子为0以确保结果可复现
random.seed(0)  # 设置Python内置随机数生成器的种子为0
n_episode = 1  # 设置采样专家数据的回合数为1
expert_s, expert_a = sample_expert_data(n_episode)  # 调用函数采样专家的状态和动作数据

n_samples = 30  # 设置从专家数据中随机采样的样本数量为30
random_index = random.sample(range(expert_s.shape[0]), n_samples)  # 随机选择30个索引
expert_s = expert_s[random_index]  # 根据随机索引选择对应的专家状态数据
expert_a = expert_a[random_index]  # 根据随机索引选择对应的专家动作数据



# 定义行为克隆算法类
class BehaviorClone:
    """行为克隆算法

    通过监督学习的方式，让智能体学习专家的行为策略。
    使用最大似然估计来训练策略网络。

    Args:
        state_dim (int): 状态空间维度
        hidden_dim (int): 隐藏层维度
        action_dim (int): 动作空间维度
        lr (float): 学习率
    """

    def __init__(self, state_dim, hidden_dim, action_dim, lr):
        """初始化行为克隆算法

        Args:
            state_dim (int): 状态空间维度
            hidden_dim (int): 隐藏层维度
            action_dim (int): 动作空间维度
            lr (float): 学习率
        """
        self.policy = PolicyNet(state_dim, hidden_dim, action_dim).to(device)  # 创建策略网络实例并移动到指定计算设备
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)  # 创建Adam优化器用于更新策略网络参数

    def learn(self, states, actions):
        """学习专家数据

        Args:
            states: 专家状态数据
            actions: 专家动作数据
        """
        states = torch.tensor(states, dtype=torch.float).to(device)  # 将状态数据转换为浮点型张量并移动到计算设备
        actions = torch.tensor(actions, dtype=torch.long).view(-1, 1).to(device)  # 将动作数据转换为长整型张量，重塑为列向量并移动到设备
        log_probs = torch.log(self.policy(states).gather(1, actions))  # 计算策略网络输出概率的对数，使用gather函数选择对应动作的概率
        bc_loss = torch.mean(-log_probs)  # 计算负对数似然损失（最大似然估计的负值）

        self.optimizer.zero_grad()  # 清零优化器中的梯度缓存
        bc_loss.backward()  # 反向传播计算损失函数关于网络参数的梯度
        self.optimizer.step()  # 使用计算得到的梯度更新网络参数

    def take_action(self, state):
        """根据当前状态选择动作

        Args:
            state: 当前环境状态

        Returns:
            int: 选择的动作
        """
        state = torch.tensor([state], dtype=torch.float).to(device)  # 将状态转换为浮点型张量并添加批次维度，移动到计算设备
        probs = self.policy(state)  # 通过策略网络前向传播获取动作概率分布
        action_dist = torch.distributions.Categorical(probs)  # 创建分类分布对象用于采样
        action = action_dist.sample()  # 从概率分布中采样一个动作
        return action.item()  # 将张量转换为Python整数并返回


# 定义智能体测试函数
def test_agent(agent, env, n_episode):
    """测试智能体的性能

    Args:
        agent: 要测试的智能体
        env: 测试环境
        n_episode (int): 测试回合数

    Returns:
        float: 平均回报
    """
    # 初始化回报列表
    return_list = []
    # 循环测试指定回合数
    for episode in range(n_episode):
        # 初始化回合回报
        episode_return = 0
        # 重置环境，获取初始状态
        state, _ = env.reset()  # gymnasium返回(observation, info)
        # 初始化结束标志
        done = False
        # 在一个回合内循环测试
        while not done:
            # 使用智能体选择动作
            action = agent.take_action(state)
            # 执行动作，获取下一状态和奖励
            next_state, reward, done, truncated, _ = env.step(action)  # gymnasium返回5个值
            # 更新当前状态
            state = next_state
            # 累加回报
            episode_return += reward
            # 检查是否因为时间限制而结束
            if truncated:
                done = True
        # 保存回合回报
        return_list.append(episode_return)
    # 返回平均回报
    return np.mean(return_list)


env.reset(seed=0)  # 重置环境并设置随机种子为0（gymnasium的新语法）
torch.manual_seed(0)  # 设置PyTorch的随机种子为0以确保结果可复现
np.random.seed(0)  # 设置numpy的随机种子为0以确保结果可复现

lr = 1e-3  # 设置行为克隆算法的学习率为0.001
bc_agent = BehaviorClone(state_dim, hidden_dim, action_dim, lr)  # 创建行为克隆智能体实例
n_iterations = 500  # 设置训练迭代次数为200（减少用于快速测试）
batch_size = 64  # 设置每次训练的批次大小为64
test_returns = []  # 初始化空列表用于存储测试回报

with tqdm(total=n_iterations, desc="进度条") as pbar:  # 创建进度条对象用于显示训练进度
    for i in range(n_iterations):  # 循环进行指定次数的训练迭代
        sample_indices = np.random.randint(low=0,
                                           high=expert_s.shape[0],
                                           size=batch_size)  # 随机生成批次大小的索引用于采样专家数据
        bc_agent.learn(expert_s[sample_indices], expert_a[sample_indices])  # 使用采样的专家数据训练行为克隆智能体
        current_return = test_agent(bc_agent, env, 5)  # 测试当前智能体的性能，运行5个回合
        test_returns.append(current_return)  # 将测试回报添加到列表中
        if (i + 1) % 10 == 0:  # 每10次迭代更新一次进度条显示
            pbar.set_postfix({'return': '%.3f' % np.mean(test_returns[-10:])})  # 显示最近10次测试的平均回报
        pbar.update(1)  # 更新进度条



