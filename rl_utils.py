# 导入所需的库
from tqdm import tqdm  # 用于显示进度条
import numpy as np    # 用于数值计算
import torch         # PyTorch深度学习库
import collections   # 用于创建双端队列
import random       # 用于随机采样


class ReplayBuffer:
    """经验回放池，用于存储和采样训练数据

    这个类实现了一个固定大小的经验回放缓冲区，用于存储和随机采样转换数据
    (state, action, reward, next_state, done)。使用collections.deque作为底层数据结构。

    Attributes:
        buffer: 一个双端队列，用于存储转换数据
    """

    def __init__(self, capacity):
        """初始化经验回放池

        Args:
            capacity: int, 回放池的最大容量
        """
        # collections.deque是一个双端队列，maxlen参数限制其最大长度
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done): 
        """添加一条转换记录到回放池中

        Args:
            state: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            next_state: 下一个状态
            done: 是否结束
        """
        # 将转换元组添加到缓冲区
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size): 
        """从回放池中随机采样一个批次的数据

        Args:
            batch_size: int, 采样的批次大小

        Returns:
            tuple: 包含state, action, reward, next_state, done的批次数据
        """
        # random.sample从buffer中随机采样batch_size个转换
        transitions = random.sample(self.buffer, batch_size)
        # zip(*transitions)将转换列表解压成单独的状态、动作等列表
        state, action, reward, next_state, done = zip(*transitions)
        # 返回numpy数组格式的状态和下一状态，其他保持原格式
        return np.array(state), action, reward, np.array(next_state), done

    def size(self): 
        """返回当前回放池中的样本数量

        Returns:
            int: 回放池中的样本数量
        """
        return len(self.buffer)


def moving_average(a, window_size):
    """计算数组的移动平均值

    使用滑动窗口计算数组的移动平均值，包括开始和结束部分的特殊处理。

    Args:
        a: np.array, 输入数组
        window_size: int, 移动平均的窗口大小

    Returns:
        np.array: 移动平均后的数组
    """
    # np.insert在数组a的开始处插入0
    cumulative_sum = np.cumsum(np.insert(a, 0, 0))
    # 计算中间部分的移动平均
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    # 生成用于计算开始部分的索引序列
    r = np.arange(1, window_size-1, 2)
    # 计算开始部分的移动平均
    begin = np.cumsum(a[:window_size-1])[::2] / r
    # 计算结束部分的移动平均
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    # 连接开始、中间和结束部分
    return np.concatenate((begin, middle, end))


def train_on_policy_agent(env, agent, num_episodes):
    """训练在线策略智能体的函数

    针对在线策略算法（如REINFORCE、Actor-Critic等）的训练函数。

    Args:
        env: gym环境实例
        agent: 强化学习智能体实例
        num_episodes: int, 总训练回合数

    Returns:
        list: 每个回合的累积奖励列表
    """
    return_list = []  # 存储每个回合的累积奖励
    for i in range(10):  # 将总回合数分成10个部分
        # 使用tqdm创建进度条
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/10)):
                episode_return = 0  # 当前回合的累积奖励
                # 创建用于存储转换数据的字典
                transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
                state, _ = env.reset()  # 重置环境，获取初始状态
                done = False  # 回合是否结束的标志
                while not done:  # 一个回合的循环
                    action = agent.take_action(state)
                    # gymnasium的step方法返回5个值
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated  # 合并两种终止情况
                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(done)
                    state = next_state
                    episode_return += reward
                return_list.append(episode_return)
                agent.update(transition_dict)
                if (i_episode+1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1),
                                    'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    return return_list


def train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimal_size, batch_size):
    """训练离线策略智能体的函数

    针对离线策略算法（如DQN、DDPG等）的训练函数。实现了经验回放机制，
    只有当回放池中的样本数量达到最小要求时才开始训练。

    Args:
        env: gym环境实例
        agent: 强化学习智能体实例
        num_episodes: int, 总训练回合数
        replay_buffer: ReplayBuffer实例, 经验回放池
        minimal_size: int, 开始训练需要的最小样本数量
        batch_size: int, 每次训练的批次大小

    Returns:
        list: 每个回合的累积奖励列表
    """
    return_list = []  # 存储每个回合的累积奖励
    # 将总回合数分成10个部分，便于显示训练进度
    for i in range(10):
        # 使用tqdm创建进度条，显示训练进度
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/10)):
                episode_return = 0  # 记录当前回合的累积奖励
                state, _ = env.reset()  # 重置环境，获取初始状态
                done = False  # 回合结束标志
                # 一个回合的交互循环
                while not done:
                    # 智能体根据当前状态选择动作
                    action = agent.take_action(state)
                    # 环境交互，获取next_state, reward等信息，使用gymnasium API
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated  # 合并两种终止情况
                    # 将transition存储到回放池中
                    replay_buffer.add(state, action, reward, next_state, done)
                    state = next_state  # 更新状态
                    episode_return += reward  # 累积奖励
                    # 当回放池中样本数量达到要求时，进行训练
                    if replay_buffer.size() > minimal_size:
                        # 从回放池采样一个批次的数据
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                        # 将采样数据整理成字典格式
                        transition_dict = {
                            'states': b_s,
                            'actions': b_a,
                            'next_states': b_ns,
                            'rewards': b_r,
                            'dones': b_d
                        }
                        # 更新智能体的策略
                        agent.update(transition_dict)

                return_list.append(episode_return)  # 保存当前回合的累积奖励
                # 每10个回合更新一次进度条信息
                if (i_episode+1) % 10 == 0:
                    pbar.set_postfix({
                        'episode': '%d' % (num_episodes/10 * i + i_episode+1),
                        'return': '%.3f' % np.mean(return_list[-10:])  # 显示最近10个回合的平均回报
                    })
                pbar.update(1)  # 更新进度条
    return return_list


def compute_advantage(gamma, lmbda, td_delta):
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
