# 导入所需的库
import torch  # 导入PyTorch深度学习框架
import torch.nn.functional as F  # 导入PyTorch中的函数性工具
import numpy as np  # 导入数值计算库
import random  # 导入随机数生成库
import matplotlib.pyplot as plt  # 导入绘图库
import gym  # 导入OpenAI Gym强化学习环境
import rl_utils  # 导入自定义的强化学习工具库
from tqdm import tqdm  # 添加tqdm用于显示进度条


class Qnet(torch.nn.Module):
    """标准Q网络类，实现了一个简单的前馈神经网络。

    这个网络包含一个隐藏层，用于近似Q值函数。

    Args:
        state_dim (int): 输入的状态维度
        hidden_dim (int): 隐藏层的维度
        action_dim (int): 输出的动作维度

    Attributes:
        fc1 (torch.nn.Linear): 第一个全连接层，从状态到隐藏层
        fc2 (torch.nn.Linear): 第二个全连接层，从隐藏层到动作值
    """
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Qnet, self).__init__()
        # 第一层全连接层，将状态映射到隐藏层
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        # 第二层全连接层，将隐藏层映射到动作值
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        """前向传播函数。

        Args:
            x (torch.Tensor): 输入状态张量

        Returns:
            torch.Tensor: 每个动作的Q值
        """
        # 使用ReLU激活函数处理第一层的输出
        x = F.relu(self.fc1(x))
        # 返回最终的Q值预测
        return self.fc2(x)


class VAnet(torch.nn.Module):
    """Dueling DQN的价值优势网络类。

    实现了具有价值流(V)和优势流(A)的神经网络架构。

    Args:
        state_dim (int): 输入的状态维度
        hidden_dim (int): 隐藏层的维度
        action_dim (int): 输出的动作维度

    Attributes:
        fc1 (torch.nn.Linear): 共享的特征提取层
        fc_A (torch.nn.Linear): 优势流的全连接层
        fc_V (torch.nn.Linear): 价值流的全连接层
    """
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(VAnet, self).__init__()
        # 共享特征提取层
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        # 优势函数网络
        self.fc_A = torch.nn.Linear(hidden_dim, action_dim)
        # 价值函数网络
        self.fc_V = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        """前向传播函数。

        Args:
            x (torch.Tensor): 输入状态张量

        Returns:
            torch.Tensor: 通过价值和优势计算得到的Q值
        """
        # 提取共享特征并计算优势值
        A = self.fc_A(F.relu(self.fc1(x)))
        # 提取共享特征并计算状态值
        V = self.fc_V(F.relu(self.fc1(x)))
        # 结合优势和价值，减去平均优势以确保可识别性
        Q = V + A - A.mean(1).view(-1, 1)
        return Q


class DQN:
    """DQN算法的实现类，支持vanilla DQN、Double DQN和Dueling DQN。

    该类实现了DQN算法的核心功能，包括经验回放、目标网络更新等机制。

    Args:
        state_dim (int): 状态空间的维度
        hidden_dim (int): 神经网络隐藏层的维度
        action_dim (int): 动作空间的维度
        learning_rate (float): 学习率
        gamma (float): 折扣因子
        epsilon (float): ε-贪婪策略中的探索率
        target_update (int): 目标网络更新频率
        device (str): 运行设备（'cpu'或'cuda'）
        dqn_type (str, optional): DQN的类型，可选'VanillaDQN'、'DoubleDQN'或'DuelingDQN'

    Attributes:
        action_dim (int): 动作空间维度
        q_net (torch.nn.Module): 主Q网络
        target_q_net (torch.nn.Module): 目标Q网络
        optimizer (torch.optim.Adam): 优化器
        gamma (float): 折扣因子
        epsilon (float): 探索率
        target_update (int): 目标网络更新频率
        count (int): 更新步数计数器
        dqn_type (str): DQN类型
        device (str): 运行设备
    """
    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, gamma,
                 epsilon, target_update, device, dqn_type='VanillaDQN'):
        self.action_dim = action_dim
        # 根据DQN类型选择不同的网络架构
        if dqn_type == 'DuelingDQN':
            # 使用价值优势网络作为Q网络和目标网络
            self.q_net = VAnet(state_dim, hidden_dim, self.action_dim).to(device)
            self.target_q_net = VAnet(state_dim, hidden_dim, self.action_dim).to(device)
        else:
            # 使用标准Q网络作为Q网络和目标网络
            self.q_net = Qnet(state_dim, hidden_dim, self.action_dim).to(device)
            self.target_q_net = Qnet(state_dim, hidden_dim, self.action_dim).to(device)

        # 初始化Adam优化器
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)
        # 设置其他超参数
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # 探索率
        self.target_update = target_update  # 目标网络更新频率
        self.count = 0  # 更新次数计数器
        self.dqn_type = dqn_type  # DQN类型
        self.device = device  # 运行设备

    def take_action(self, state):
        """选择动作的函数，实现ε-贪婪策略。

        Args:
            state (numpy.ndarray): 当前状态

        Returns:
            int: 选择的动作索引
        """
        # epsilon贪婪策略：随机探索或选择最优动作
        if np.random.random() < self.epsilon:
            # 随机选择动作
            action = np.random.randint(self.action_dim)
        else:
            # 选择Q值最大的动作
            state = torch.tensor([state], dtype=torch.float).to(self.device)
            action = self.q_net(state).argmax().item()
        return action

    def max_q_value(self, state):
        """计算给定状态下的最大Q值。

        Args:
            state (numpy.ndarray): 输入状态

        Returns:
            float: 最大Q值
        """
        # 将状态转换为张量并移到指定设备
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        # 返回最大Q值
        return self.q_net(state).max().item()

    def update(self, transition_dict):
        """更新网络参数。

        使用经验回放中的样本更新Q网络参数。

        Args:
            transition_dict (dict): 包含转换样本的字典，包括：
                - 'states': 状态batch
                - 'actions': 动作batch
                - 'rewards': 奖励batch
                - 'next_states': 下一状态batch
                - 'dones': 终止标志batch
        """
        # 将所有数据转换为张量并移到指定设备
        states = torch.tensor(transition_dict['states'],
                            dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                             dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                 dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                           dtype=torch.float).view(-1, 1).to(self.device)

        # 计算当前Q值
        q_values = self.q_net(states).gather(1, actions)

        # 根据不同的DQN类型计算目标Q值
        if self.dqn_type == 'DoubleDQN':
            # Double DQN: 使用当前网络选择动作，目标网络评估动作
            max_action = self.q_net(next_states).max(1)[1].view(-1, 1)
            max_next_q_values = self.target_q_net(next_states).gather(1, max_action)
        else:
            # Vanilla DQN: 直接使用目标网络的最大Q值
            max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1, 1)

        # 计算目标Q值
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)
        # 计算均方误差损失
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))

        # 优化模型参数
        self.optimizer.zero_grad()  # 清除梯度
        dqn_loss.backward()  # 反向传播
        self.optimizer.step()  # 更新参数

        # 定期更新目标网络
        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.count += 1


def train_DQN(agent, env, num_episodes, replay_buffer, minimal_size, batch_size):
    """训练DQN智能体的函数。

    Args:
        agent (DQN): DQN智能体实例
        env (gym.Env): OpenAI Gym环境
        num_episodes (int): 训练的总回合数
        replay_buffer (ReplayBuffer): 经验回放缓冲区
        minimal_size (int): 开始训练的最小样本数
        batch_size (int): 每次训练的批量大小

    Returns:
        tuple: (return_list, max_q_value_list)
            - return_list: 每个回合的回报列表
            - max_q_value_list: 每个回合的最大Q值列表
    """
    return_list = []  # 记录每个回合的回报
    max_q_value_list = []  # 记录每个回合的最大Q值
    max_q_value = 0  # 初始化最大Q值

    # 分10次迭代训练
    for i in range(10):
        with tqdm(total=int(num_episodes / 10),
                  desc='Iteration %d' % i) as pbar:
            # 每次迭代训练一部分回合
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0  # 记录当前回合的累积回报
                state, _ = env.reset()  # 重置环境，获取初始状态
                done = False

                # 进行一回合训练
                while not done:
                    action = agent.take_action(state)  # 选择动作
                    # 平滑处理最大Q值
                    max_q_value = agent.max_q_value(state) * 0.005 + max_q_value * 0.995
                    max_q_value_list.append(max_q_value)  # 保存当前状态的最大Q值

                    # 执行动作，获得下一状态和奖励
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated  # 合并两种终止情况

                    # 存储转移数据到经验回放缓冲区
                    replay_buffer.add(state, action, reward, next_state, done)
                    state = next_state
                    episode_return += reward  # 累积回报

                    # 如果样本数量达到要求，就进行训练
                    if replay_buffer.size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                        transition_dict = {
                            'states': b_s,
                            'actions': b_a,
                            'rewards': b_r,
                            'next_states': b_ns,
                            'dones': b_d
                        }
                        agent.update(transition_dict)  # 更新智能体

                return_list.append(episode_return)  # 记录回合回报

                # 每训练10个回合，更新进度条
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({
                        'episode':
                        '%d' % (num_episodes / 10 * i + i_episode + 1),
                        'return':
                        '%.3f' % np.mean(return_list[-10:])
                    })
                pbar.update(1)  # 更新进度条

    return return_list, max_q_value_list

# 在主代码之前设置环境和超参数
env_name = 'CartPole-v1'  # 定义环境名称
env = gym.make(env_name)  # 创建环境实例
state_dim = env.observation_space.shape[0]  # 状态空间维度
hidden_dim = 128  # 隐藏层维度
action_dim = env.action_space.n  # 动作空间维度
lr = 2e-3  # 学习率
gamma = 0.98  # 折扣因子
epsilon = 0.1  # 探索率
target_update = 10  # 目标网络更新频率
buffer_size = 10000  # 经验回放池大小
minimal_size = 500  # 开始训练的最小样本数
batch_size = 64  # 批量大小
num_episodes = 200  # 训练回合数
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")  # 设备选择

# 设置随机种子以确保可重复性
random.seed(0)  # 设置Python随机数生成器的种子
np.random.seed(0)  # 设置NumPy随机数生成器的种子
torch.manual_seed(0)  # 设置PyTorch随机数生成器的种子
# 使用新版本的gym环境随机种子设置方式
env = gym.make(env_name, render_mode=None)  # 重新创建环境
env.action_space.seed(0)  # 设置动作空间的随机种子
env.observation_space.seed(0)  # 设置观察空间的随机种子
env.reset(seed=0)  # 设置环境的随机种子

# 创建经验回放缓冲区
replay_buffer = rl_utils.ReplayBuffer(buffer_size)

# 创建Dueling DQN智能体
agent = DQN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon,
            target_update, device, 'DuelingDQN')

# 训练智能体并获取训练数据
return_list, max_q_value_list = train_DQN(agent, env, num_episodes,
                                         replay_buffer, minimal_size,
                                         batch_size)

# 绘制训练回报曲线
episodes_list = list(range(len(return_list)))  # 创建回合数列表
mv_return = rl_utils.moving_average(return_list, 5)  # 计算移动平均回报
plt.plot(episodes_list, mv_return)  # 绘制回报曲线
plt.xlabel('Episodes')  # 设置x轴标签
plt.ylabel('Returns')  # 设置y轴标签
plt.title('Dueling DQN on {}'.format(env_name))  # 设置图标标题
plt.show()  # 显示图表

# 绘制Q值变化曲线
frames_list = list(range(len(max_q_value_list)))  # 创建帧数列表
plt.plot(frames_list, max_q_value_list)  # 绘制Q值曲线
plt.axhline(0, c='orange', ls='--')  # 添加y=0参考线
plt.axhline(10, c='red', ls='--')  # 添加y=10参考线
plt.xlabel('Frames')  # 设置x轴标签
plt.ylabel('Q value')  # 设置y轴标签
plt.title('Dueling DQN on {}'.format(env_name))  # 设置图表标题
plt.show()  # 显示图表
