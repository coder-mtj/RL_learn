import gymnasium as gym  # 使用gymnasium替代旧版gym，提供强化学习环境
import torch  # PyTorch深度学习框架，用于构建和训练神经网络
import torch.nn.functional as F  # PyTorch函数式接口，提供激活函数等
import torch.nn as nn  # PyTorch神经网络模块，提供损失函数等
import numpy as np  # 数值计算库，用于数组操作和数学计算
import matplotlib.pyplot as plt  # 绘图库，用于可视化训练结果
from tqdm import tqdm  # 进度条库，用于显示训练进度
import random  # Python随机数库，用于随机采样
import rl_utils  # 自定义强化学习工具模块，包含训练函数和工具类


class PolicyNet(torch.nn.Module):
    """策略网络类，用于输出动作概率分布

    这是一个简单的两层全连接神经网络，输入状态，输出各个动作的概率。
    使用ReLU激活函数和Softmax输出层。
    """

    def __init__(self, state_dim, hidden_dim, action_dim):
        """初始化策略网络

        Args:
            state_dim (int): 状态空间的维度
            hidden_dim (int): 隐藏层的神经元数量
            action_dim (int): 动作空间的维度
        """
        super(PolicyNet, self).__init__()  # 调用父类构造函数
        # 第一层：从状态维度到隐藏层维度的线性变换
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        # 第二层：从隐藏层维度到动作维度的线性变换
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        """前向传播函数

        Args:
            x (torch.Tensor): 输入的状态张量，形状为 (batch_size, state_dim)

        Returns:
            torch.Tensor: 动作概率分布，形状为 (batch_size, action_dim)
        """
        # 通过第一层并应用ReLU激活函数
        x = F.relu(self.fc1(x))
        # 通过第二层并应用Softmax激活函数，得到概率分布
        return F.softmax(self.fc2(x), dim=1)


class ValueNet(torch.nn.Module):
    """价值网络类，用于估计状态价值函数m

    这是一个简单的两层全连接神经网络，输入状态，输出该状态的价值估计。
    用于PPO算法中的Critic部分。
    """

    def __init__(self, state_dim, hidden_dim):
        """初始化价值网络

        Args:
            state_dim (int): 状态空间的维度
            hidden_dim (int): 隐藏层的神经元数量
        """
        super(ValueNet, self).__init__()  # 调用父类构造函数
        # 第一层：从状态维度到隐藏层维度的线性变换
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        # 第二层：从隐藏层维度到1维输出（价值标量）
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        """前向传播函数

        Args:
            x (torch.Tensor): 输入的状态张量，形状为 (batch_size, state_dim)

        Returns:
            torch.Tensor: 状态价值估计，形状为 (batch_size, 1)
        """
        # 通过第一层并应用ReLU激活函数
        x = F.relu(self.fc1(x))
        # 通过第二层，输出价值估计（无激活函数）
        return self.fc2(x)


class PPO:
    """PPO（Proximal Policy Optimization）算法实现类

    采用截断方式的PPO算法，包含Actor-Critic架构。
    Actor网络用于策略学习，Critic网络用于价值函数估计。
    """

    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                 lmbda, epochs, eps, gamma, device):
        """初始化PPO算法

        Args:
            state_dim (int): 状态空间维度
            hidden_dim (int): 神经网络隐藏层维度
            action_dim (int): 动作空间维度
            actor_lr (float): Actor网络学习率
            critic_lr (float): Critic网络学习率
            lmbda (float): GAE（广义优势估计）中的lambda参数
            epochs (int): 每次更新时的训练轮数
            eps (float): PPO截断参数，控制策略更新幅度
            gamma (float): 折扣因子
            device (torch.device): 计算设备（CPU或GPU）
        """
        # 创建Actor网络（策略网络）并移动到指定设备
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        # 创建Critic网络（价值网络）并移动到指定设备
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        # 创建Actor网络的优化器，使用Adam算法
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        # 创建Critic网络的优化器，使用Adam算法
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)
        # 折扣因子，用于计算未来奖励的现值
        self.gamma = gamma
        # GAE中的lambda参数，用于平衡偏差和方差
        self.lmbda = lmbda
        # 每次更新时使用同一批数据训练的轮数
        self.epochs = epochs
        # PPO截断参数，防止策略更新过大
        self.eps = eps
        # 存储计算设备信息
        self.device = device

    def take_action(self, state):
        """根据当前状态选择动作

        Args:
            state (np.ndarray): 当前环境状态

        Returns:
            int: 选择的动作索引
        """
        # 将numpy状态转换为PyTorch张量，并添加batch维度
        # 使用numpy.array确保输入是单个数组而不是数组列表
        state = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)
        # 通过Actor网络获取动作概率分布
        probs = self.actor(state)
        # 创建分类分布对象，用于采样动作
        action_dist = torch.distributions.Categorical(probs)
        # 从概率分布中采样一个动作
        action = action_dist.sample()
        # 返回动作的整数值
        return action.item()

    def update(self, transition_dict):
        """更新PPO算法的Actor和Critic网络

        Args:
            transition_dict (dict): 包含状态转换数据的字典，包含：
                - states: 状态列表
                - actions: 动作列表
                - rewards: 奖励列表
                - next_states: 下一状态列表
                - dones: 回合结束标志列表
        """
        # 将状态数据转换为张量并移动到设备
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)
        # 将动作数据转换为张量，reshape为列向量
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(
            self.device)
        # 将奖励数据转换为张量，reshape为列向量
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        # 将下一状态数据转换为张量并移动到设备
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)
        # 将回合结束标志转换为张量，reshape为列向量
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)
        # 计算TD目标值：r + γ * V(s') * (1 - done)
        td_target = rewards + self.gamma * self.critic(next_states) * (1 -
                                                                       dones)
        # 计算TD误差：TD目标值 - 当前状态价值估计
        td_delta = td_target - self.critic(states)
        # 使用GAE计算优势函数值
        advantage = rl_utils.compute_advantage(self.gamma, self.lmbda,
                                               td_delta.cpu()).to(self.device)
        # 计算旧策略的对数概率，用于重要性采样比率计算
        old_log_probs = torch.log(self.actor(states).gather(1,
                                                            actions)).detach()

        # 进行多轮训练更新
        for _ in range(self.epochs):
            # 计算当前策略的对数概率
            log_probs = torch.log(self.actor(states).gather(1, actions))
            # 计算重要性采样比率：π_new / π_old
            ratio = torch.exp(log_probs - old_log_probs)
            # 计算未截断的代理目标
            surr1 = ratio * advantage
            # 计算截断的代理目标，限制策略更新幅度
            surr2 = torch.clamp(ratio, 1 - self.eps,
                                1 + self.eps) * advantage
            # 计算Actor损失：取两个代理目标的最小值的负数
            actor_loss = torch.mean(-torch.min(surr1, surr2))
            # 计算Critic损失：TD目标值与价值估计的均方误差
            critic_loss = torch.mean(
                F.mse_loss(self.critic(states), td_target.detach()))
            # 清零Actor优化器的梯度
            self.actor_optimizer.zero_grad()
            # 清零Critic优化器的梯度
            self.critic_optimizer.zero_grad()
            # 反向传播计算Actor梯度
            actor_loss.backward()
            # 反向传播计算Critic梯度
            critic_loss.backward()
            # 更新Actor网络参数
            self.actor_optimizer.step()
            # 更新Critic网络参数
            self.critic_optimizer.step()


# PPO算法的超参数设置
actor_lr = 1e-3  # Actor网络的学习率
critic_lr = 1e-2  # Critic网络的学习率
num_episodes = 250  # 预训练PPO的回合数
hidden_dim = 128  # 神经网络隐藏层维度
gamma = 0.98  # 折扣因子，用于计算未来奖励的现值
lmbda = 0.95  # GAE中的lambda参数，平衡偏差和方差
epochs = 10  # 每次更新时的训练轮数
eps = 0.2  # PPO截断参数，控制策略更新幅度
# 自动选择计算设备：优先使用GPU，否则使用CPU
device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
    "cpu")

# 环境设置和PPO预训练
env_name = 'CartPole-v1'  # 使用CartPole-v1环境，v0版本已过时
env = gym.make(env_name)  # 创建gym环境实例
# 设置环境随机种子，确保结果可重现（使用新API）
env.reset(seed=0)
# 设置PyTorch随机种子，确保神经网络初始化一致
torch.manual_seed(0)
# 获取状态空间维度（观测空间的形状）
state_dim = env.observation_space.shape[0]
# 获取动作空间维度（离散动作的数量）
action_dim = env.action_space.n
# 创建PPO智能体实例，用于预训练生成专家数据
ppo_agent = PPO(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda,
                epochs, eps, gamma, device)

# 使用PPO算法进行预训练，生成专家策略
return_list = rl_utils.train_on_policy_agent(env, ppo_agent, num_episodes)



def sample_expert_data(n_episode):
    """采样专家数据的函数

    Args:
        n_episode: int, 采样的回合数

    Returns:
        tuple: 专家状态和动作的numpy数组
    """
    states = []  # 存储状态
    actions = []  # 存储动作
    for episode in range(n_episode):
        # 重置环境，获取初始状态（新API返回状态和信息）
        state, _ = env.reset()
        done = False  # 回合结束标志
        while not done:
            # 使用训练好的PPO智能体选择动作
            action = ppo_agent.take_action(state)
            states.append(state)  # 保存当前状态
            actions.append(action)  # 保存选择的动作
            # 执行动作，获取下一状态（新API返回5个值）
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated  # 合并两种终止条件
            state = next_state  # 更新状态
    return np.array(states), np.array(actions)  # 返回numpy数组格式


# 设置随机种子确保结果可重现
env.reset(seed=0)  # 使用新API设置环境种子
torch.manual_seed(0)  # 设置PyTorch随机种子
random.seed(0)  # 设置Python随机种子
n_episode = 1  # 采样1个回合的专家数据
expert_s, expert_a = sample_expert_data(n_episode)  # 获取专家状态和动作

n_samples = 30  # 从专家数据中采样30个样本
random_index = random.sample(population=(expert_s.shape[0]), k=n_samples)  # 随机选择索引
expert_s = expert_s[random_index]  # 获取采样的专家状态
expert_a = expert_a[random_index]  # 获取采样的专家动作


class Discriminator(nn.Module):
    """判别器网络，用于区分专家数据和智能体数据

    判别器接收状态和动作作为输入，输出一个概率值，
    表示输入数据来自专家的概率。
    """
    def __init__(self, state_dim, hidden_dim, action_dim):
        """初始化判别器网络

        Args:
            state_dim: int, 状态空间维度
            hidden_dim: int, 隐藏层维度
            action_dim: int, 动作空间维度
        """
        super(Discriminator, self).__init__()
        # 第一层：输入为状态和动作的拼接
        self.fc1 = torch.nn.Linear(state_dim + action_dim, hidden_dim)
        # 第二层：输出单个概率值
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x, a):
        """前向传播

        Args:
            x: torch.Tensor, 状态张量
            a: torch.Tensor, 动作张量（one-hot编码）

        Returns:
            torch.Tensor: 输出概率值
        """
        # 将状态和动作拼接
        cat = torch.cat([x, a], dim=1)
        # 通过第一层并应用ReLU激活
        x = F.relu(self.fc1(cat))
        # 通过第二层并应用Sigmoid激活，输出概率
        # tanh是将函数值映射到(-1, 1)之间，而sigmoid则是将函数值映射到(0, 1)之间，从而作为概率
        return torch.sigmoid(self.fc2(x))

class GAIL:
    """生成对抗模仿学习（GAIL）算法实现

    GAIL通过对抗训练的方式，让智能体学习模仿专家的行为。
    判别器用于区分专家数据和智能体数据，智能体则试图欺骗判别器。
    """
    def __init__(self, agent, state_dim, action_dim, hidden_dim, lr_d):
        """初始化GAIL算法

        Args:
            agent: 强化学习智能体（如PPO）
            state_dim: int, 状态空间维度
            action_dim: int, 动作空间维度
            hidden_dim: int, 判别器隐藏层维度
            lr_d: float, 判别器学习率
        """
        # 创建判别器网络并移动到指定设备
        self.discriminator = Discriminator(state_dim, hidden_dim,
                                           action_dim).to(device)
        # 创建判别器优化器
        self.discriminator_optimizer = torch.optim.Adam(
            self.discriminator.parameters(), lr=lr_d)
        # 保存智能体引用
        self.agent = agent

    def learn(self, expert_s, expert_a, agent_s, agent_a, next_s, dones):
        """GAIL学习过程

        Args:
            expert_s: 专家状态数据
            expert_a: 专家动作数据
            agent_s: 智能体状态数据
            agent_a: 智能体动作数据
            next_s: 下一状态数据
            dones: 回合结束标志
        """
        # 将数据转换为张量并移动到设备
        expert_states = torch.tensor(expert_s, dtype=torch.float).to(device)
        # 将专家动作转换为长整型张量（one_hot要求LongTensor）
        expert_actions = torch.tensor(expert_a, dtype=torch.long).to(device)
        agent_states = torch.tensor(agent_s, dtype=torch.float).to(device)
        # 将智能体动作转换为长整型张量（one_hot要求LongTensor）
        agent_actions = torch.tensor(agent_a, dtype=torch.long).to(device)
        # 将动作转换为one-hot编码（必须使用LongTensor作为输入）
        expert_actions = F.one_hot(expert_actions, num_classes=2).float()
        agent_actions = F.one_hot(agent_actions, num_classes=2).float()

        # 计算判别器对专家和智能体数据的输出概率
        expert_prob = self.discriminator(expert_states, expert_actions)
        agent_prob = self.discriminator(agent_states, agent_actions)
        # 计算判别器损失：希望对智能体数据输出1，对专家数据输出0
        discriminator_loss = nn.BCELoss()(
            agent_prob, torch.ones_like(agent_prob)) + nn.BCELoss()(
                expert_prob, torch.zeros_like(expert_prob))
        # 更新判别器
        self.discriminator_optimizer.zero_grad()
        discriminator_loss.backward()
        self.discriminator_optimizer.step()

        # 计算智能体的奖励：-log(D(s,a))，鼓励智能体欺骗判别器
        rewards = -torch.log(agent_prob).detach().cpu().numpy()
        # 构建转换字典用于更新智能体
        transition_dict = {
            'states': agent_s,
            'actions': agent_a,
            'rewards': rewards,
            'next_states': next_s,
            'dones': dones
        }
        # 更新智能体策略
        self.agent.update(transition_dict)


# GAIL训练阶段的初始化设置
# 重新设置环境随机种子，确保GAIL训练的可重现性
env.reset(seed=0)
# 重新设置PyTorch随机种子，确保网络初始化一致
torch.manual_seed(0)
# 设置判别器的学习率
lr_d = 1e-3
# 创建新的PPO智能体实例用于GAIL训练（重新初始化，不使用预训练权重）
agent = PPO(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda,
            epochs, eps, gamma, device)
# 创建GAIL算法实例，传入智能体和网络参数
gail = GAIL(agent, state_dim, action_dim, hidden_dim, lr_d)
# 设置GAIL训练的总回合数
n_episode = 500
# 初始化列表，用于存储每个回合的累积奖励
return_list = []

# 使用进度条显示训练进度
with tqdm(total=n_episode, desc="GAIL训练进度") as pbar:
    for i in range(n_episode):
        episode_return = 0  # 当前回合累积奖励
        # 重置环境，获取初始状态（新API）
        state, _ = env.reset()
        done = False  # 回合结束标志
        # 存储当前回合的转换数据
        state_list = []
        action_list = []
        next_state_list = []
        done_list = []
        # 一个回合的交互循环
        while not done:
            # 智能体选择动作
            action = agent.take_action(state)
            # 执行动作，获取环境反馈（新API）
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated  # 合并两种终止条件
            # 保存转换数据
            state_list.append(state)
            action_list.append(action)
            next_state_list.append(next_state)
            done_list.append(done)
            state = next_state  # 更新状态
            episode_return += reward  # 累积奖励
        # 保存当前回合的累积奖励
        return_list.append(episode_return)
        # 使用GAIL进行学习更新
        gail.learn(expert_s, expert_a, state_list, action_list,
                   next_state_list, done_list)
        # 每10回合更新一次进度条显示
        if (i + 1) % 10 == 0:
            pbar.set_postfix({'return': '%.3f' % np.mean(return_list[-10:])})
        pbar.update(1)  # 更新进度条


# 绘制训练结果图表
iteration_list = list(range(len(return_list)))  # 创建回合数列表
plt.plot(iteration_list, return_list)  # 绘制累积奖励曲线
plt.xlabel('Episodes')  # 设置x轴标签
plt.ylabel('Returns')  # 设置y轴标签
plt.title('GAIL on {}'.format(env_name))  # 设置图表标题
plt.show()  # 显示图表