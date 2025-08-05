# 导入所需的库
import gymnasium as gym  # 使用gymnasium替代旧版gym，提供强化学习环境
import torch  # PyTorch深度学习框架
import torch.nn.functional as F  # PyTorch的函数式接口，包含激活函数等
import numpy as np  # 数值计算库
import matplotlib.pyplot as plt  # 绘图库
from tqdm import tqdm  # 用于显示训练进度条


class PolicyNetContinuous(torch.nn.Module):
    """连续动作空间的策略网络

    该网络输出高斯分布的均值和标准差，用于连续动作空间的策略。

    这里我们不能直接生成连续的值，而是通过正态分布进行模拟。这里mu也就是均值，我们将mu限制在[-2, 2]以内从而控制其值域

    Args:
        state_dim (int): 状态空间的维度
        hidden_dim (int): 隐藏层的维度
        action_dim (int): 动作空间的维度
    """
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNetContinuous, self).__init__()  # 调用父类构造函数
        # 第一层全连接层，从状态维度到隐藏维度
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        # 第二层全连接层，从隐藏维度到动作维度（均值）
        self.fc_mu = torch.nn.Linear(hidden_dim, action_dim)
        # 第三层全连接层，从隐藏维度到动作维度（标准差）
        self.fc_std = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        """前向传播函数

        Args:
            x (torch.Tensor): 输入状态张量

        Returns:
            tuple: (mu, std) 高斯分布的均值和标准差
        """
        # 使用ReLU激活函数处理第一层的输出
        x = F.relu(self.fc1(x))
        # 计算动作分布的均值，使用tanh限制在[-2, 2]范围内
        mu = 2.0 * torch.tanh(self.fc_mu(x))
        # 计算动作分布的标准差，使用softplus确保为正值
        std = F.softplus(self.fc_std(x))
        return mu, std  # 返回高斯分布的均值和标准差


class ValueNet(torch.nn.Module):
    """价值网络(Critic)，用于评估状态的价值

    该网络将状态映射为标量值，表示该状态的价值估计。
    使用两层全连接神经网络实现。

    Args:
        state_dim (int): 状态空间的维度
        hidden_dim (int): 隐藏层的维度
    """
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()  # 调用父类构造函数
        # 第一层全连接层，从状态维度到隐藏维度
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        # 第二层全连接层，从隐藏维度到1（输出单个价值）
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        """前向传播函数

        Args:
            x (torch.Tensor): 输入状态张量

        Returns:
            torch.Tensor: 状态价值估计
        """
        # 使用ReLU激活函数处理第一层的输出
        x = F.relu(self.fc1(x))
        # 输出状态价值（无需激活函数）
        return self.fc2(x)


class PPOContinuous:
    """PPO算法的连续动作空间实现

    该算法通过限制策略更新的幅度来提高训练稳定性。
    采用截断方式来约束新旧策略之间的比率，防止策略更新过大。

    Args:
        state_dim (int): 状态空间的维度
        hidden_dim (int): 隐藏层的维度
        action_dim (int): 动作空间的维度
        actor_lr (float): 策略网络的学习率
        critic_lr (float): 价值网络的学习率
        lmbda (float): GAE参数，用于优势函数计算
        epochs (int): 每次更新的训练轮数
        eps (float): PPO截断参数
        gamma (float): 折扣因子
        device (torch.device): 计算设备
    """
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                 lmbda, epochs, eps, gamma, device):
        # 创建策略网络并移动到指定设备
        self.actor = PolicyNetContinuous(state_dim, hidden_dim,
                                         action_dim).to(device)
        # 创建价值网络并移动到指定设备
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        # 为策略网络创建Adam优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        # 为价值网络创建Adam优化器
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)
        # 保存折扣因子，用于计算未来奖励的现值
        self.gamma = gamma
        # 保存GAE参数，用于优势函数计算
        self.lmbda = lmbda
        # 保存训练轮数，一条序列的数据用来训练的轮数
        self.epochs = epochs
        # 保存PPO截断参数，控制策略更新幅度
        self.eps = eps
        # 保存计算设备
        self.device = device

    def take_action(self, state):
        """根据当前状态选择动作

        使用策略网络输出动作概率分布，然后从中采样一个动作。

        通过state和actor函数获取mu以及std，然后在合成一个分布。通过分布进行随机抽样

        Args:
            state: np.array, 当前环境状态

        Returns:
            int: 选择的动作索引
        """
        # 将状态转换为张量并添加批次维度，然后移动到设备
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        # 通过策略网络获取动作概率分布
        mu, std = self.actor(state)
        # 创建正态分布对象
        action_dist = torch.distributions.Normal(mu, std)
        # 从分布中采样动作
        action = action_dist.sample()
        # 返回动作值（转换为numpy数组格式，适合gym环境）
        return action.detach().cpu().numpy().flatten()


    def update(self, transition_dict):
        """更新策略网络和价值网络

        使用PPO算法更新网络参数。计算优势函数，然后使用截断的重要性采样
        比率来更新策略网络，同时更新价值网络以更好地估计状态价值。

        Args:
            transition_dict: dict, 包含训练所需的各种数据
        """
        # 将numpy数组转换为PyTorch张量并移动到设备
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)

        # 计算TD目标值：r + γ * V(s') * (1 - done)
        # 通过时分差别进行计算TD目标值，我们要做的就是让critic输出的值接近此值
        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        # 计算TD误差：TD目标值 - 当前状态价值估计
        # 计算其中的delta，以为计算GAE要使用
        td_delta = td_target - self.critic(states)

        # 使用GAE计算优势函数，先移动到CPU计算再移回设备
        # 计算广义优势函数，因为TRPO、PPO公式都需要用到
        advantage = self.compute_advantage(self.gamma, self.lmbda,
                                      td_delta.cpu()).to(self.device)
        # 计算旧策略的对数概率并分离梯度（不参与反向传播）
        mu, std = self.actor(states)

        # 计算旧策略的动作分布和对数概率
        # 这里实际上是随机初始化的一个旧策略
        old_action_dists = torch.distributions.Normal(mu.detach(),
                                                      std.detach())
        # 计算旧策略下动作的对数概率
        # 后面方便计算ratio
        old_log_probs = old_action_dists.log_prob(actions).detach()

        # 进行多轮训练更新
        for _ in range(self.epochs):
            # 计算当前策略的对数概率
            mu, std = self.actor(states)
            # 创建正态分布对象
            action_dists = torch.distributions.Normal(mu, std)
            # 计算新策略下动作的对数概率
            # 这里计算回溯中当前动作的概率
            log_probs = action_dists.log_prob(actions)
            # 计算重要性采样比率
            # 计算ratio
            ratio = torch.exp(log_probs - old_log_probs)
            # 计算未截断的代理目标
            # 计算PPO公式的左边
            surr1 = ratio * advantage
            # 计算截断的代理目标
            # clip计算PPO公式的右边
            surr2 = torch.clamp(ratio, 1 - self.eps,
                                1 + self.eps) * advantage
            # 计算策略损失
            # actor的损失是梯度下降的，所以采用相反的PPO-penalty
            actor_loss = -torch.mean(torch.min(surr1, surr2))
            # 计算价值网络损失
            critic_loss = torch.mean(
                F.mse_loss(self.critic(states), td_target.detach()))
            # 梯度更新
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()

    def compute_advantage(self, gamma, lmbda, td_delta):
        """计算广义优势估计（GAE）

        使用GAE（Generalized Advantage Estimation）方法计算优势函数值。
        这是一种结合了n步优势和TD误差的方法，用于减小策略梯度的方差。

        Args:
            gamma: float, 折扣因子，用于平衡未来和当前的回报
            lmbda: float, GAE的平滑参数，用于权衡偏差和方差
            td_delta: torch.Tensor, 时间差分误差

        Returns:
            torch.Tensor: 计算得到的优势函数值
        """
        td_delta = td_delta.detach().numpy()
        advantage_list = []
        advantage = 0.0
        for delta in td_delta[::-1]:
            advantage = gamma * lmbda * advantage + delta
            advantage_list.append(advantage)
        advantage_list.reverse()
        return torch.tensor(advantage_list, dtype=torch.float)


# ==================== 超参数设置 ====================
# 策略网络学习率，控制策略更新的步长
actor_lr = 1e-3
# 价值网络学习率，通常设置得比策略网络大一些
critic_lr = 1e-2
# 总训练回合数
num_episodes = 500
# 神经网络隐藏层维度
hidden_dim = 128
# 折扣因子，用于计算未来奖励的现值，接近1表示更重视长期奖励
gamma = 0.98
# GAE参数，用于优势函数计算，权衡偏差和方差
lmbda = 0.95
# 每次更新时的训练轮数，PPO会重复使用同一批数据训练多轮
epochs = 10
# PPO截断参数，控制策略更新幅度，防止更新过大
eps = 0.2
# 自动选择计算设备：如果有GPU则使用GPU，否则使用CPU
device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
    "cpu")

# ==================== 环境设置 ====================
# 环境名称，Pendulum-v1是经典的连续控制任务（倒立摆）
env_name = 'Pendulum-v1'
# 创建环境实例
env = gym.make(env_name)
# 设置环境随机种子以确保结果可重现
env.reset(seed=0)
# 设置PyTorch随机种子以确保神经网络初始化的可重现性
torch.manual_seed(0)
# 获取状态空间维度（Pendulum环境中为3维：cos(θ), sin(θ), 角速度）
state_dim = env.observation_space.shape[0]
# 获取动作空间维度（Pendulum环境中为1维：扭矩值）
action_dim = env.action_space.shape[0]

# ==================== 智能体创建 ====================
# 创建PPO智能体实例，传入所有必要的参数
agent = PPOContinuous(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda,
            epochs, eps, gamma, device)

# ==================== 开始训练 ====================
# 初始化存储每个回合累积奖励的列表
return_list = []



# 分10个阶段进行训练，便于观察训练进度
for i in range(10):
    # 创建进度条，显示当前阶段的训练进度
    with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
        # 在当前阶段进行 num_episodes/10 个回合的训练
        for i_episode in range(int(num_episodes/10)):
            # 初始化当前回合的累积奖励
            episode_return = 0
            # 创建用于存储转换数据的字典
            transition_dict = {
                'states': [],      # 存储状态序列
                'actions': [],     # 存储动作序列
                'next_states': [], # 存储下一状态序列
                'rewards': [],     # 存储奖励序列
                'dones': []        # 存储结束标志序列
            }

            # 重置环境，获取初始状态（gymnasium返回状态和信息）
            state, _ = env.reset()
            # 初始化回合结束标志
            done = False

            # 开始一个回合的交互循环
            while not done:
                # 智能体根据当前状态选择动作
                action = agent.take_action(state)
                # 环境执行动作，返回下一状态、奖励等信息
                # gymnasium的step方法返回5个值：next_state, reward, terminated, truncated, info
                next_state, reward, terminated, truncated, _ = env.step(action)
                # 合并两种终止情况：terminated（任务完成）或truncated（超时）
                done = terminated or truncated

                # 将当前转换存储到字典中
                transition_dict['states'].append(state)           # 当前状态
                transition_dict['actions'].append(action)         # 执行的动作
                transition_dict['next_states'].append(next_state) # 下一状态
                transition_dict['rewards'].append(reward)         # 获得的奖励
                transition_dict['dones'].append(done)             # 是否结束

                # 更新状态为下一状态，准备下一步交互
                state = next_state
                # 累积当前回合的奖励
                episode_return += reward

            # 将当前回合的累积奖励添加到列表中
            return_list.append(episode_return)
            # 使用收集到的转换数据更新智能体的策略和价值网络
            agent.update(transition_dict)

            # 每10个回合更新一次进度条的显示信息
            if (i_episode + 1) % 10 == 0:
                # 设置进度条后缀，显示当前回合数和最近10个回合的平均回报
                pbar.set_postfix({
                    'episode': '%d' % (num_episodes/10 * i + i_episode + 1),  # 总回合数
                    'return': '%.3f' % np.mean(return_list[-10:])              # 最近10回合平均回报
                })
            # 更新进度条
            pbar.update(1)


