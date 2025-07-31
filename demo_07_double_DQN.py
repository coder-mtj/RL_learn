# 导入必要的库
import random  # 用于生成随机数
import gym     # 强化学习环境库
import numpy as np  # 数值计算库
import torch   # 深度学习框架
import torch.nn.functional as F  # PyTorch的函数库
import matplotlib.pyplot as plt  # 绘图库
import rl_utils  # 自定义的强化学习工具库
from tqdm import tqdm  # 进度条显示库


class Qnet(torch.nn.Module):
    '''Q网络，用于近似Q函数

    一个简单的前馈神经网络，包含一个隐藏层，用于估计状态动作值函数(Q值)

    Args:
        state_dim (int): 状态空间维度
        hidden_dim (int): 隐藏层神经元数量
        action_dim (int): 动作空间维度
    '''
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Qnet, self).__init__()
        # 第一层全连接层，从状态维度到隐藏层维度
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        # 第二层全连接层，从隐藏层维度到动作维度
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        """前向传播函数

        Args:
            x (torch.Tensor): 输入状态

        Returns:
            torch.Tensor: 每个动作对应的Q值
        """
        # 使用ReLU激活函数
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class DQN:
    '''DQN算法的实现，包括常规DQN和Double DQN

    Args:
        state_dim (int): 状态空间维度
        hidden_dim (int): 隐藏层神经元数量
        action_dim (int): 动作空间维度
        learning_rate (float): 学习率
        gamma (float): 折扣因子
        epsilon (float): epsilon-贪婪策略中的探索概率
        target_update (int): 目标网络更新频率
        device (torch.device): 运行设备(CPU/GPU)
        dqn_type (str): DQN类型，可选'VanillaDQN'或'DoubleDQN'
    '''
    def __init__(self,
                 state_dim,
                 hidden_dim,
                 action_dim,
                 learning_rate,
                 gamma,
                 epsilon,
                 target_update,
                 device,
                 dqn_type='VanillaDQN'):
        # 初始化DQN智能体的各项参数
        self.action_dim = action_dim  # 动作空间维度
        # 创建在线Q网络并移动到指定设备
        self.q_net = Qnet(state_dim, hidden_dim, self.action_dim).to(device)
        # 创建目标Q网络并移动到指定设备
        self.target_q_net = Qnet(state_dim, hidden_dim,
                                 self.action_dim).to(device)
        # 使用Adam优化器
        self.optimizer = torch.optim.Adam(self.q_net.parameters(),
                                          lr=learning_rate)
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # 探索概率
        self.target_update = target_update  # 目标网络更新频率
        self.count = 0  # 用于记录更新次数
        self.dqn_type = dqn_type  # DQN类型
        self.device = device  # 运行设备

    def take_action(self, state):
        """选择动作的函数

        使用epsilon-贪婪策略选择动作

        Args:
            state (np.ndarray): 当前状态

        Returns:
            int: 选择的动作
        """
        # epsilon-贪婪策略
        if np.random.random() < self.epsilon:
            # 随机探索
            action = np.random.randint(self.action_dim)
        else:
            # 利用当前策略
            state = torch.tensor([state], dtype=torch.float).to(self.device)
            action = self.q_net(state).argmax().item()
        return action

    def max_q_value(self, state):
        """计算当前状态下的最大Q值

        Args:
            state (np.ndarray): 当前状态

        Returns:
            float: 最大Q值
        """
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        return self.q_net(state).max().item()

    def update(self, transition_dict):
        """更新神经网络参数

        Args:
            transition_dict (dict): 包含'states', 'actions', 'rewards', 'next_states', 'dones'的转移字典
        """
        # 将所有数据转换为张量并移动到指定设备
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(
            self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)

        # 计算当前状态的Q值
        q_values = self.q_net(states).gather(1, actions)

        # 根据算法类型选择不同的目标Q值计算方式
        if self.dqn_type == 'DoubleDQN':
            # Double DQN: 使用在线网络选择动作，目标网络估计Q值
            max_action = self.q_net(next_states).max(1)[1].view(-1, 1)
            max_next_q_values = self.target_q_net(next_states).gather(1, max_action)
        else:
            # 普通DQN: 直接使用目标网络计算最大Q值
            max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1, 1)

        # 计算目标Q值
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)
        # 计算TD误差
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))
        # 梯度清零
        self.optimizer.zero_grad()
        # 反向传播
        dqn_loss.backward()
        # 更新参数
        self.optimizer.step()

        # 定期更新目标网络
        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(
                self.q_net.state_dict())
        self.count += 1

# 超参数设置
lr = 1e-2  # 学习率
num_episodes = 200  # 训练回合数
hidden_dim = 128  # 隐藏层神经元数
gamma = 0.98  # 折扣因子
epsilon = 0.01  # 探索概率
target_update = 50  # 目标网络更新频率
buffer_size = 5000  # 经验回放池大小
minimal_size = 1000  # 开始学习的最小经验池大小
batch_size = 64  # 批量大小

# 设置运行设备（GPU如果可用，否则使用CPU）
device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
    "cpu")

# 创建并初始化环境
env_name = 'Pendulum-v1'
env = gym.make(env_name, render_mode=None)
state_dim = env.observation_space.shape[0]  # 状态空间维度
action_dim = 11  # 将连续动作分成11个离散动作


def dis_to_con(discrete_action, env, action_dim):
    """将离散动作转换为连续动作

    由于Pendulum环境本身是连续动作空间，而DQN处理离散动作空间，
    需要将DQN输出的离散动作转换回连续动作。

    Args:
        discrete_action (int): 离散动作索引(0 到 action_dim-1)
        env (gym.Env): 强化学习环境，用于获取动作空间的范围
        action_dim (int): 离散动作空间的大小

    Returns:
        float: 转换后的连续动作值
    """
    action_lowbound = env.action_space.low[0]  # 连续动作的最小值
    action_upbound = env.action_space.high[0]  # 连续动作的最大值
    # 将离散动作索引按比例映射到连续动作空间
    return action_lowbound + (discrete_action /
                              (action_dim - 1)) * (action_upbound -
                                                   action_lowbound)


def train_DQN(agent, env, num_episodes, replay_buffer, minimal_size,
              batch_size):
    """训练DQN智能体

    Args:
        agent (DQN): DQN智能体
        env (gym.Env): 强化学习环境
        num_episodes (int): 训练回合数
        replay_buffer (rl_utils.ReplayBuffer): 经验回放池
        minimal_size (int): 开始学习的最小经验池大小
        batch_size (int): 批量大小

    Returns:
        list, list: 每个回合的累计奖励和每个状态的最大Q值
    """
    return_list = []  # 存储每个回合的累计奖励
    max_q_value_list = []  # 存储每个状态的最大Q值
    max_q_value = 0
    # 分10次迭代训练
    for i in range(10):
        with tqdm(total=int(num_episodes / 10),
                  desc='Iteration %d' % i) as pbar:
            # 每次迭代训练一部分回合
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0  # 当前回合的累计奖励
                state, _ = env.reset()  # 更新reset返回值的解包
                done = False
                # 进行一回合训练
                while not done:
                    action = agent.take_action(state)  # 选择动作
                    # 平滑处理最大Q值
                    max_q_value = agent.max_q_value(
                        state) * 0.005 + max_q_value * 0.995
                    max_q_value_list.append(max_q_value)  # 保存每个状态的最大Q值
                    action_continuous = dis_to_con(action, env,
                                                   agent.action_dim)  # 离散动作转连续动作
                    # 更新环境状态
                    next_state, reward, terminated, truncated, _ = env.step([action_continuous])  # 更新step返回值的解包
                    done = terminated or truncated
                    # 将经历的转移存入经验回放池
                    replay_buffer.add(state, action, reward, next_state, done)
                    state = next_state
                    episode_return += reward  # 更新累计奖励
                    # 当经验回放池中的经验足够时，开始更新智能体
                    if replay_buffer.size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(
                            batch_size)  # 从经验回放池中随机采样
                        transition_dict = {
                            'states': b_s,
                            'actions': b_a,
                            'next_states': b_ns,
                            'rewards': b_r,
                            'dones': b_d
                        }
                        agent.update(transition_dict)  # 更新智能体
                return_list.append(episode_return)  # 记录当前回合的累计奖励
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


# ============== 训练普通DQN智能体 ==============
# 设置随机种子以确保实验可重复性
random.seed(0)  # 设置Python内置random模块的随机种子
np.random.seed(0)  # 设置NumPy的随机种子
torch.manual_seed(0)  # 设置PyTorch的随机种子，影响神经网络的初始化和随机操作

# 创建经验回放池，用于存储和采样训练数据
# buffer_size在之前定义为5000，表示最多存储5000个转移样本
replay_buffer = rl_utils.ReplayBuffer(buffer_size)

# 初始化DQN智能体
# 使用前面定义的超参数：state_dim（状态维度）, hidden_dim（隐藏层大小）, action_dim（动作维度）
# lr（学习率）, gamma（折扣因子）, epsilon（探索率）, target_update（目标网络更新频率）
agent = DQN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon,
            target_update, device)  # device决定是用CPU还是GPU训练

# 训练DQN智能体
# train_DQN函数返回两个列表：累计奖励列表和最大Q值列表
# minimal_size=1000表示经验池中至少有1000个样本才开始学习
# batch_size=64表示每次从经验池采样64个样本进行学习
return_list, max_q_value_list = train_DQN(agent, env, num_episodes,
                                          replay_buffer, minimal_size,
                                          batch_size)

# ============== 绘制DQN训练结果 ==============
# 绘制累计奖励曲线
episodes_list = list(range(len(return_list)))  # 创建episode索引列表
mv_return = rl_utils.moving_average(return_list, 5)  # 计算5回合的移动平均，使曲线更平滑
# 使用matplotlib绘制曲线
plt.plot(episodes_list, mv_return)  # plot()函数绘制曲线，x轴为回合数，y轴为平均回报
plt.xlabel('Episodes')  # 设置x轴标签
plt.ylabel('Returns')   # 设置y轴标签
plt.title('DQN on {}'.format(env_name))  # 设置图表标题，format()方法用于字符串格式化
plt.show()  # 显示图表

# 绘制Q值变化曲线
frames_list = list(range(len(max_q_value_list)))  # 创建帧索引列表
plt.plot(frames_list, max_q_value_list)  # 绘制Q值随时间变化的曲线
# 添加水平参考线，ls='--'表示虚线样式
plt.axhline(0, c='orange', ls='--')  # 在y=0处添加橙色虚线
plt.axhline(10, c='red', ls='--')    # 在y=10处添加红色虚线
plt.xlabel('Frames')  # 设置x轴标签为帧数
plt.ylabel('Q value')  # 设置y轴标签为Q值
plt.title('DQN on {}'.format(env_name))  # 设置图表标题
plt.show()  # 显示图表

# ============== 训练Double DQN智能体 ==============
# 重新设置随机种子，确保与DQN实验的初始条件相同
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

# 创建新的经验回放池
replay_buffer = rl_utils.ReplayBuffer(buffer_size)

# 初始化Double DQN智能体
# 与前面的DQN相比，最后多了一个参数'DoubleDQN'，指定使用Double DQN算法
# Double DQN通过分离动作选择和评估来减少Q值过估计问题
agent = DQN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon,
            target_update, device, 'DoubleDQN')

# 训练Double DQN智能体
# 使用相同的训练函数和参数，但是智能体的更新逻辑会不同
return_list, max_q_value_list = train_DQN(agent, env, num_episodes,
                                          replay_buffer, minimal_size,
                                          batch_size)

# ============== 可视化Double DQN的训练结果 ==============
# 绘制Double DQN累计奖励曲线
episodes_list = list(range(len(return_list)))  # 生成回合数列表，用于x轴
# 对奖励进行移动平均处理，window_size=5表示每5个回合取一次平均
# 这样可以使曲线更平滑，更容易观察训练趋势
mv_return = rl_utils.moving_average(return_list, 5)

# 使用matplotlib绘制训练曲线
plt.plot(episodes_list, mv_return)  # 绘制回合数-平均奖励曲线
plt.xlabel('Episodes')  # x轴标注为回合数
plt.ylabel('Returns')   # y轴标注为累计奖励
# 使用字符串格式化设置图表标题，显示环境名称
plt.title('Double DQN on {}'.format(env_name))
plt.show()  # 展示图表

# 绘制Double DQN的Q值变化曲线
# 生成帧数列表，用于x轴。每一帧对应一个状态转换
frames_list = list(range(len(max_q_value_list)))
plt.plot(frames_list, max_q_value_list)  # 绘制帧数-Q值曲线

# 添加两条水平参考线，用于比较Q值的范围
plt.axhline(0, c='orange', ls='--')   # 在y=0处添加橙色虚线
plt.axhline(10, c='red', ls='--')     # 在y=10处添加红色虚线，用于观察Q值是否过估计

plt.xlabel('Frames')   # x轴标注为帧数
plt.ylabel('Q value')  # y轴标注为Q值
# 设置图表标题，标明是Double DQN的结果
plt.title('Double DQN on {}'.format(env_name))
plt.show()  # 显示图表

# 至此，我们完成了DQN和Double DQN在Pendulum环境下的训练和性能对比
# 可以通过比较两个算法的回报曲线和Q值曲线来分析它们的性能差异
# Double DQN通常能够缓解DQN中的Q值过估计问题，产生更稳定的训练过程
