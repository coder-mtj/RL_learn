# 导入所需的库
import gym  # 导入OpenAI Gym强化学习环境
import torch  # 导入PyTorch深度学习框架
import torch.nn.functional as F  # 导入PyTorch中的函数性工具
import numpy as np  # 导入数值计算库
import matplotlib.pyplot as plt  # 导入绘图库
from tqdm import tqdm  # 导入进度条库
import rl_utils  # 导入自定义的强化学习工具库
import random  # 导入随机数生成库


class PolicyNet(torch.nn.Module):
    """策略网络类，用于REINFORCE算法。

    实现了一个简单的前馈神经网络，用于近似策略函数。

    Args:
        state_dim (int): 输入的状态维度
        hidden_dim (int): 隐藏层的维度
        action_dim (int): 输出的动作维度

    Attributes:
        fc1 (torch.nn.Linear): 第一个全连接层
        fc2 (torch.nn.Linear): 第二个全连接层，输出动作概率
    """
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        # 第一层全连接层，从状态到隐藏层
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        # 第二层全连接层，从隐藏层到动作概率
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        """前向传播函数。

        Args:
            x (torch.Tensor): 输入状态张量

        Returns:
            torch.Tensor: 各个动作的概率分布
        """
        # 使用ReLU激活函数处理第一层的输出
        x = F.relu(self.fc1(x))
        # 使用softmax函数输出动作概率
        return F.softmax(self.fc2(x), dim=1)


class REINFORCE:
    """REINFORCE算法的实现类。

    实现了基于策略梯度的REINFORCE算法，包括策略网络、动作采样和参数更新等功能。

    Args:
        state_dim (int): 状态空间的维度
        hidden_dim (int): 神经网络隐藏层的维度
        action_dim (int): 动作空间的维度
        learning_rate (float): 学习率
        gamma (float): 折扣因子
        device (str): 运行设备（'cpu'或'cuda'）

    Attributes:
        policy_net (PolicyNet): 策略网络
        optimizer (torch.optim.Adam): 优化器
        gamma (float): 折扣因子
        device (str): 运行设备
    """
    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, gamma, device):
        # 创建策略网络并移动到指定设备
        self.policy_net = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        # 使用Adam优化器
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.gamma = gamma  # 折扣因子
        self.device = device  # 设备类型

    def take_action(self, state):
        """根据当前状态采样动作。

        使用策略网络计算动作概率分布，然后进行采样。

        Args:
            state (numpy.ndarray): 当前状态

        Returns:
            int: 选择的动作
        """
        # 将状态转换为张量并移到指定设备
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        # 计算动作概率分布
        probs = self.policy_net(state)
        # 创建分类分布
        '''
            这里将probs转换为torch.distributions.Categorical对象，
            该对象表示一个多项分布，probs是每个动作的概率
            从而随机从这个分布中采样动作。
            Categorical分布的sample方法会根据给定的概率分布随机选择一个
        '''
        action_dist = torch.distributions.Categorical(probs)
        # 从分布中采样动作
        action = action_dist.sample()
        return action.item()

    def update(self, transition_dict):
        """更新策略网络参数。

        使用蒙特卡洛方法计算回报，并更新策略网络。

        Args:
            transition_dict (dict): 包含轨迹数据的字典，包括：
                - 'states': 状态列表
                - 'actions': 动作列表
                - 'rewards': 奖励列表
        """
        reward_list = transition_dict['rewards']  # 获取奖励列表
        state_list = transition_dict['states']    # 获取状态列表
        action_list = transition_dict['actions']  # 获取动作列表

        G = 0  # 初始化回报
        self.optimizer.zero_grad()  # 清除梯度
        # 从后往前计算每个时间步的损失
        for i in reversed(range(len(reward_list))):
            reward = reward_list[i]  # 获取当前步奖励
            state = torch.tensor([state_list[i]], dtype=torch.float).to(self.device)
            action = torch.tensor([action_list[i]]).view(-1, 1).to(self.device)
            # 计算对数概率
            log_prob = torch.log(self.policy_net(state).gather(1, action))
            # 计算折扣回报
            G = self.gamma * G + reward
            # 计算策略梯度损失
            loss = -log_prob * G
            # 反向传播计算梯度
            loss.backward()
        # 执行梯度下降
        self.optimizer.step()

# 设置超参数
learning_rate = 1e-3  # 学习率
num_episodes = 1000   # 训练回合数
hidden_dim = 128      # 隐藏层维度
gamma = 0.98         # 折扣因子
# 设置运行设备（GPU/CPU）
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# 创建和配置环境
env_name = "CartPole-v1"  # 使用v1版本的环境
env = gym.make(env_name, render_mode=None)  # 创建环境实例

# 设置随机种子
random.seed(0)  # 设置Python随机数生成器的种子
np.random.seed(0)  # 设置Numpy随机数生成器的种子
torch.manual_seed(0)  # 设置PyTorch随机数生成器的种子
env.action_space.seed(0)  # 设置动作空间的随机种子
env.observation_space.seed(0)  # 设置观察空间的随机种子
state, _ = env.reset(seed=0)  # 重置环境并设置随机种子

# 获取环境的状态和动作空间维度
state_dim = env.observation_space.shape[0]  # 状态空间维度
action_dim = env.action_space.n  # 动作空间维度

# 创建REINFORCE智能体
agent = REINFORCE(state_dim, hidden_dim, action_dim, learning_rate, gamma, device)

# 记录训练过程中的回报
return_list = []

# 将训练过程分成10次迭代
for i in range(10):
    # 使用tqdm创建进度条
    with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
        # 每次迭代训练num_episodes/10个回合
        for i_episode in range(int(num_episodes / 10)):
            episode_return = 0  # 记录当前回合的累积回报
            # 创建用于存储轨迹数据的字典
            transition_dict = {
                'states': [],       # 存储状态
                'actions': [],      # 存储动作
                'next_states': [],  # 存储下一状态
                'rewards': [],      # 存储奖励
                'dones': []         # 存储终止标志
            }

            state, _ = env.reset()  # 重置环境，获取初始状态
            done = False  # 回合终止标志

            # 一个回合的交互循环
            while not done:
                action = agent.take_action(state)  # 选择动作
                # 执行动作，获得下一状态、奖励和是否终止的信息
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated  # 合并两种终止情况
                # 存储轨迹数据
                transition_dict['states'].append(state)
                transition_dict['actions'].append(action)
                transition_dict['next_states'].append(next_state)
                transition_dict['rewards'].append(reward)
                transition_dict['dones'].append(done)

                state = next_state  # 更新状态
                episode_return += reward  # 累积奖励

            return_list.append(episode_return)  # 记录回合总回报
            agent.update(transition_dict)  # 使用当前回合的数据更新策略网络

            # 每训练10个回合，更新一次进度条信息
            if (i_episode + 1) % 10 == 0:
                pbar.set_postfix({
                    'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                    'return': '%.3f' % np.mean(return_list[-10:])  # 显示最近10个回合的平均回报
                })
            pbar.update(1)  # 更新进度条

# 绘制训练学习曲线
episodes_list = list(range(len(return_list)))  # 创建回合数列表
mv_return = rl_utils.moving_average(return_list, 9)  # 计算移动平均回报
plt.plot(episodes_list, mv_return)  # 绘制回报曲线
plt.xlabel('Episodes')  # 设置x轴标签
plt.ylabel('Returns')  # 设置y轴标签
plt.title('REINFORCE on {}'.format(env_name))  # 设置图表标题
plt.show()  # 显示图表
