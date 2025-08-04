import random
import gymnasium as gym  # 统一使用gymnasium库
import numpy as np
import collections
from tqdm import tqdm  # 进度条显示
import torch
import torch.nn.functional as F  # 神经网络函数库
import matplotlib.pyplot as plt
import rl_utils  # 自定义工具库（需提前下载或克隆）


class ReplayBuffer:
    ''' 经验回放池 '''

    def __init__(self, capacity):
        # 使用双端队列实现，设定最大容量（当队列满时自动丢弃最早的数据）
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        # 将单步交互数据(state, action, reward, next_state, done)存入缓冲区
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        # 随机采样batch_size条数据
        transitions = random.sample(self.buffer, batch_size)
        # 将采样的数据拆分成独立的数组（zip*是解压操作）
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    def size(self):
        # 返回当前缓冲区中的数据量
        return len(self.buffer)


class Qnet(torch.nn.Module):
    ''' Q网络：两层全连接神经网络 '''

    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Qnet, self).__init__()
        # 第一层：状态输入 -> 隐藏层
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        # 第二层：隐藏层 -> 动作输出
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        # 前向传播：状态输入 -> ReLU激活 -> 动作价值输出
        x = F.relu(self.fc1(x))  # 隐藏层使用ReLU激活
        return self.fc2(x)  # 输出层不激活（直接作为Q值）


class DQN:
    ''' DQN算法实现 '''

    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, gamma,
                 epsilon, target_update, device):
        self.action_dim = action_dim
        # 主Q网络（实时更新）
        self.q_net = Qnet(state_dim, hidden_dim, action_dim).to(device)
        # 目标Q网络（延迟更新）
        self.target_q_net = Qnet(state_dim, hidden_dim, action_dim).to(device)
        # 同步主网络参数到目标网络
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        # 使用Adam优化器
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)
        self.gamma = gamma  # 折扣因子（未来奖励衰减系数）
        self.epsilon = epsilon  # ε-贪婪策略中的探索率
        self.target_update = target_update  # 目标网络更新频率（每隔多少步同步一次）
        self.count = 0  # 计数器（记录主网络更新次数）
        self.device = device  # 计算设备（CPU/GPU）

    def take_action(self, state):
        ''' ε-贪婪策略选择动作 '''
        if np.random.random() < self.epsilon:
            # 探索：随机选择动作
            action = np.random.randint(self.action_dim)
        else:
            # 利用：选择当前状态下Q值最大的动作
            state = torch.tensor([state], dtype=torch.float).to(self.device)
            action = self.q_net(state).argmax().item()  # 取最大Q值的动作索引
        return action

    def update(self, transition_dict):
        ''' 更新Q网络参数 '''
        # 从字典中提取数据并转换为Tensor
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)


        '''
            Q网络计算Q，目标网络计算估算值r+γ * max Q'（Q-learning）
            损失函数就是这俩L2损失
        '''
        # 计算当前状态动作对的Q值（使用主网络）
        # net运算之后其shape为[1, action_dim]
        # gather函数(dim, index)表示按照指定的dim读取其index指向的数据
        q_values = self.q_net(states).gather(1, actions)  # gather按actions索引取值
        # 计算下一状态的最大Q值（使用目标网络）
        # 同上shape为[1, action_dim]，故调用max(1)沿dim=1进行计算最大值然后得到[1, 1]shape的向量
        # .[0]读取出来，使用view()改变形状，同reshape的作用
        max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1, 1)
        # 计算TD目标：r + γ * max Q(s',a') * (1 - done)
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)
        # 计算均方误差损失
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))

        # 梯度清零 -> 反向传播 -> 参数更新
        self.optimizer.zero_grad()
        dqn_loss.backward()
        self.optimizer.step()

        # 定期同步主网络参数到目标网络
        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.count += 1


# 超参数设置
lr = 2e-3  # 学习率
num_episodes = 500  # 总训练轮数
hidden_dim = 128  # Q网络隐藏层维度
gamma = 0.98  # 折扣因子
epsilon = 0.01  # 探索率
target_update = 10  # 目标网络更新频率
buffer_size = 10000  # 经验回放池容量
minimal_size = 500  # 最小训练数据量（缓冲区达到该值后才开始训练）
batch_size = 64  # 每次训练的样本量
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")  # 自动选择设备

# 创建环境并设置随机种子（保证可重复性）
env_name = 'CartPole-v1'  # 更新为v1版本
env = gym.make(env_name)
# random.seed(0)
# np.random.seed(0)
# torch.manual_seed(0)
# # 使用新的随机种子方法
# env.reset(seed=0)


# 初始化经验回放池和DQN智能体
replay_buffer = ReplayBuffer(buffer_size)
state_dim = env.observation_space.shape[0]  # 状态维度（CartPole为4）
action_dim = env.action_space.n  # 动作维度（CartPole为2）
agent = DQN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon, target_update, device)

# 训练过程
return_list = []  # 记录每轮 episode 的总回报
for i in range(10):  # 分10个阶段训练（每个阶段50轮）
    with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
        for i_episode in range(int(num_episodes / 10)):
            episode_return = 0  # 当前轮次的总回报
            state, _ = env.reset(seed=0)  # 重置环境，获取初始状态
            done = False
            while not done:
                # 选择动作 -> 执行动作 -> 存储数据
                action = agent.take_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)  # 更新为新API
                done = terminated or truncated  # 合并两个终止条件
                replay_buffer.add(state, action, reward, next_state, done)
                state = next_state
                episode_return += reward

                # 当缓冲区数据量足够时开始训练
                if replay_buffer.size() > minimal_size:
                    # 采样一批数据并转换为字典格式
                    b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                    transition_dict = {
                        'states': b_s,
                        'actions': b_a,
                        'next_states': b_ns,
                        'rewards': b_r,
                        'dones': b_d
                    }
                    agent.update(transition_dict)  # 更新Q网络

            return_list.append(episode_return)
            # 每10轮显示一次平均回报
            if (i_episode + 1) % 10 == 0:
                pbar.set_postfix({
                    'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                    'return': '%.3f' % np.mean(return_list[-10:])
                })
            pbar.update(1)

# 绘制回报曲线
episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('DQN on {}'.format(env_name))
plt.show()

# 绘制滑动平均回报曲线（平滑曲线）
mv_return = rl_utils.moving_average(return_list, 9)  # 窗口大小为9的滑动平均
plt.plot(episodes_list, mv_return)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('DQN on {} (Moving Average)'.format(env_name))
plt.show()