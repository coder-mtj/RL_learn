# 导入必要的库
import gym  # 导入OpenAI Gym强化学习环境
import torch  # 导入PyTorch深度学习框架
import torch.nn.functional as F  # 导入PyTorch中的函数性工具
import numpy as np  # 导入NumPy数值计算库
import matplotlib.pyplot as plt  # 导入Matplotlib绘图库

class PolicyNet(torch.nn.Module):
    """Actor网络，用于生成动作的概率分布。

    这个网络接收状态作为输入，输出动作的概率分布。使用两层全连接神经网络实现。

    Args:
        state_dim (int): 状态空间的维度
        hidden_dim (int): 隐藏层的维度
        action_dim (int): 动作空间的维度

    Attributes:
        fc1 (torch.nn.Linear): 第一个全连接层
        fc2 (torch.nn.Linear): 第二个全连接层
    """
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()  # 调用父类的初始化方法
        # 第一个全连接层，从状态维度到隐藏维度
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        # 第二个全连接层，从隐藏维度到动作维度
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        """前向传播函数。

        Args:
            x (torch.Tensor): 输入状态

        Returns:
            torch.Tensor: 动作的概率分布
        """
        # 使用ReLU激活函数处理第一层的输出
        x = F.relu(self.fc1(x))
        # 使用Softmax函数将输出转换为概率分布
        return F.softmax(self.fc2(x), dim=1)

class ValueNet(torch.nn.Module):
    """Critic网络，用于评估状态的价值。

    这个网络接收状态作为输入，输出该状态的价值估计。使用两层全连接神经网络实现。

    ！！！注意这个是状态的价值估计函数，而非状态动作价值估计函数。

    Args:
        state_dim (int): 状态空间的维度
        hidden_dim (int): 隐藏层的维度

    Attributes:
        fc1 (torch.nn.Linear): 第一个全连接层
        fc2 (torch.nn.Linear): 第二个全连接层
    """
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()  # 调用父类的初始化方法
        # 第一个全连接层，从状态维度到隐藏维度
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        # 第二个全连接层，从隐藏维度到1（价值估计）
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        """前向传播函数。

        Args:
            x (torch.Tensor): 输入状态

        Returns:
            torch.Tensor: 状态的价值估计
        """
        # 使用ReLU激活函数处理第一层的输出
        x = F.relu(self.fc1(x))
        # 输出状态的价值估计
        return self.fc2(x)

class ActorCritic:
    """Actor-Critic算法的实现类。

    结合策略网络(Actor)和价值网络(Critic)的优势Actor-Critic算法实现。

    Args:
        state_dim (int): 状态空间的维度
        hidden_dim (int): 隐藏层的维度
        action_dim (int): 动作空间的维度
        actor_lr (float): Actor网络的学习率
        critic_lr (float): Critic网络的学习率
        gamma (float): 折扣因子
        device (torch.device): 运行设备（CPU/GPU）

    Attributes:
        actor (PolicyNet): Actor网络实例
        critic (ValueNet): Critic网络实例
        actor_optimizer (torch.optim.Adam): Actor网络的优化器
        critic_optimizer (torch.optim.Adam): Critic网络的优化器
    """
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                 gamma, device):
        # 创建Actor网络并移动到指定设备
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        # 创建Critic网络并移动到指定设备
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        # 创建Actor网络的Adam优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                              lr=actor_lr)
        # 创建Critic网络的Adam优化器
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                               lr=critic_lr)
        # 设置折扣因子
        self.gamma = gamma
        # 设置运行设备
        self.device = device

    def take_action(self, state):
        """选择动作的函数。

        Args:
            state (numpy.ndarray): 当前状态

        Returns:
            int: 选择的动作
        """
        # 处理新版本gym返回的(state, info)元组
        if isinstance(state, tuple):
            state = state[0]
        # 将状态转换为张量并移动到指定设备
        state = torch.FloatTensor(state).view(1, -1).to(self.device)
        # 通过Actor网络计算动作概率
        probs = self.actor(state)
        # 创建类别分布
        action_dist = torch.distributions.Categorical(probs)
        # 从分布中采样动作
        action = action_dist.sample()
        # 返回选择的动作
        return action.item()

    def update(self, transition_dict):
        """更新Actor和Critic网络的函数。

        Args:
            transition_dict (dict): 包含训练所需数据的字典，包括states, actions, rewards等
        """
        # 将状态转换为张量并移动到指定设备
        states = torch.tensor(transition_dict['states'],
                            dtype=torch.float).to(self.device)
        # 将动作转换为张量并改变形状
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(
            self.device)
        # 将奖励转换为张量并改变形状
        rewards = torch.tensor(transition_dict['rewards'],
                             dtype=torch.float).view(-1, 1).to(self.device)
        # 将下一个状态转换为张量并移动到指定设备
        next_states = torch.tensor(transition_dict['next_states'],
                                 dtype=torch.float).to(self.device)
        # 将终止标志转换为张量并改变形状
        dones = torch.tensor(transition_dict['dones'],
                           dtype=torch.float).view(-1, 1).to(self.device)

        # 计算时序差分目标
        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        # 计算时序差分误差
        td_delta = td_target - self.critic(states)
        # 计算对数概率
        log_probs = torch.log(self.actor(states).gather(1, actions))
        # 计算Actor的损失函数
        actor_loss = torch.mean(-log_probs * td_delta.detach())
        # 计算Critic的损失函数（均方误差）
        critic_loss = torch.mean(
            F.mse_loss(self.critic(states), td_target.detach()))

        # 清空Actor和Critic的梯度
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        # 反向传播计算Actor的梯度
        actor_loss.backward()
        # 反向传播计算Critic的梯度
        critic_loss.backward()
        # 更新Actor的参数
        self.actor_optimizer.step()
        # 更新Critic的参数
        self.critic_optimizer.step()

def train_on_policy_agent(env, agent, num_episodes):
    """训练在线策略智能体的函数

    Args:
        env: gym环境实例
        agent: 强化学习智能体实例
        num_episodes (int): 总训练回合数

    Returns:
        list: 每个回合的累积奖励列表
    """
    from tqdm import tqdm  # 导入进度条库
    return_list = []  # 存储每个回合的累积奖励

    for i in range(10):  # 将总回合数分成10个部分
        with tqdm(total=int(num_episodes/10), desc=f'Iteration {i}') as pbar:
            for i_episode in range(int(num_episodes/10)):
                episode_return = 0  # 当前回合的累积奖励
                transition_dict = {
                    'states': [],
                    'actions': [],
                    'next_states': [],
                    'rewards': [],
                    'dones': []
                }

                state, _ = env.reset()  # 重置环境，获取初始状态
                done = False  # 回合是否结束的标志

                while not done:  # 一个回合的交互循环
                    action = agent.take_action(state)  # 选择动作
                    # 执行动作，获取环境反馈
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated  # 合并两种终止状态
                    # 收集训练数据
                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(done)

                    state = next_state  # 更新状态
                    episode_return += reward  # 累积奖励

                return_list.append(episode_return)  # 保存当前回合的累积奖励
                agent.update(transition_dict)  # 更新智能体

                if (i_episode + 1) % 10 == 0:  # 每10个回合更新一次进度条
                    pbar.set_postfix({
                        'episode': f'{num_episodes/10 * i + i_episode+1}',
                        'return': f'{np.mean(return_list[-10:]):.3f}'
                    })
                pbar.update(1)

    return return_list

def moving_average(a, window_size):
    """计算数组的移动平均值

    Args:
        a (np.array): 输入数组
        window_size (int): 移动平均的窗口大小

    Returns:
        np.array: 移动平均后的数组
    """
    cumulative_sum = np.cumsum(np.insert(a, 0, 0))  # 计算累积和
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size-1, 2)
    begin = np.cumsum(a[:window_size-1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))

# 设置超参数
actor_lr = 1e-3  # Actor网络的学习率
critic_lr = 1e-2  # Critic网络的学习率
num_episodes = 1000  # 训练的回合数
hidden_dim = 128  # 隐藏层的维度
gamma = 0.98  # 折扣因子
# 设置运行设备（如果有GPU则使用GPU，否则使用CPU）
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# 创建并设置环境
env_name = 'CartPole-v1'  # 环境名称
env = gym.make(env_name, render_mode=None)  # 创建环境实例

# 设置随机种子以确保实验可重现
torch.manual_seed(0)  # 设置PyTorch的随机种子
np.random.seed(0)  # 设置NumPy的随机种子
state, _ = env.reset(seed=0)  # 重置环境并设置环境的随机种子

# 创建Actor-Critic智能体
state_dim = env.observation_space.shape[0]  # 状态空间维度
action_dim = env.action_space.n  # 动作空间维度
agent = ActorCritic(state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                    gamma, device)

# 训练智能体
return_list = train_on_policy_agent(env, agent, num_episodes)

# 绘制结果
episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('Actor-Critic on {}'.format(env_name))
plt.show()

# 计算移动平均并绘制
mv_return = moving_average(return_list, 9)
plt.plot(episodes_list, mv_return)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('Actor-Critic on {}'.format(env_name))
plt.show()

# 关闭环境
env.close()
