# 导入随机数生成库，用于设置随机种子和随机采样
import random
# 导入gymnasium库，这是gym的维护版本，用于创建强化学习环境
import gymnasium as gym  # 使用gymnasium替代已弃用的gym
# 导入numpy库，用于数值计算和数组操作
import numpy as np
# 导入tqdm库，用于显示训练进度条
from tqdm import tqdm
# 导入PyTorch深度学习框架
import torch
# 导入PyTorch的函数式接口，包含激活函数、损失函数等
import torch.nn.functional as F
# 导入正态分布类，用于策略网络的随机采样
from torch.distributions import Normal
# 导入matplotlib绘图库，用于绘制训练曲线
import matplotlib.pyplot as plt
# 导入自定义的强化学习工具模块
import rl_utils



# 定义连续动作空间的策略网络类，继承自PyTorch的nn.Module
class PolicyNetContinuous(torch.nn.Module):
    """连续动作空间的策略网络

    这个网络输出动作的均值和标准差，用于生成连续动作。
    使用tanh激活函数将动作限制在有界范围内。

    Args:
        state_dim (int): 状态空间的维度
        hidden_dim (int): 隐藏层的神经元数量
        action_dim (int): 动作空间的维度
        action_bound (float): 动作的边界值
    """

    def __init__(self, state_dim, hidden_dim, action_dim, action_bound):
        """初始化策略网络的各个层

        Args:
            state_dim (int): 状态空间的维度
            hidden_dim (int): 隐藏层的神经元数量
            action_dim (int): 动作空间的维度
            action_bound (float): 动作的边界值
        """
        # 调用父类的初始化方法
        super(PolicyNetContinuous, self).__init__()
        # 创建第一个全连接层，从状态维度映射到隐藏层维度
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        # 创建输出动作均值的全连接层
        self.fc_mu = torch.nn.Linear(hidden_dim, action_dim)
        # 创建输出动作标准差的全连接层
        self.fc_std = torch.nn.Linear(hidden_dim, action_dim)
        # 保存动作的边界值，用于缩放输出动作
        self.action_bound = action_bound

    def forward(self, x):
        """前向传播函数，计算给定状态下的动作和对数概率

        Args:
            x (torch.Tensor): 输入的状态张量

        Returns:
            tuple: (action, log_prob) 动作和对应的对数概率
        """
        # 通过第一个全连接层并应用ReLU激活函数
        x = F.relu(self.fc1(x))
        # 计算动作的均值
        mu = self.fc_mu(x)
        # 计算动作的标准差，使用softplus确保为正值
        std = F.softplus(self.fc_std(x))
        # 创建正态分布对象
        dist = Normal(mu, std)
        # 使用重参数化技巧进行采样，保证梯度可以反向传播
        # 这样使得采样得到的结果可导
        normal_sample = dist.rsample()  # rsample()是重参数化采样
        # 计算采样值的对数概率
        # 直接就是log(prob)
        log_prob = dist.log_prob(normal_sample)
        # 使用tanh函数将动作限制在(-1, 1)范围内
        action = torch.tanh(normal_sample)
        # 计算tanh变换后的对数概率密度，需要进行雅可比行列式修正
        # ⚠️ 与标准公式差异：应该使用torch.tanh(normal_sample)而不是torch.tanh(action)
        # 标准公式：log_prob = log_prob - torch.log(1 - torch.tanh(normal_sample).pow(2) + 1e-7)
        # 当前实现：torch.tanh(action) = torch.tanh(torch.tanh(normal_sample))，数学上不等价
        log_prob = log_prob - torch.log(1 - torch.tanh(action).pow(2) + 1e-7)
        # 将动作缩放到实际的动作边界范围
        action = action * self.action_bound
        # 返回最终的动作和对数概率
        return action, log_prob


# 定义连续动作空间的Q值网络类，继承自PyTorch的nn.Module
class QValueNetContinuous(torch.nn.Module):
    """连续动作空间的Q值网络

    这个网络接受状态和动作作为输入，输出对应的Q值。
    用于评估在给定状态下执行特定动作的价值。

    Args:
        state_dim (int): 状态空间的维度
        hidden_dim (int): 隐藏层的神经元数量
        action_dim (int): 动作空间的维度
    """

    def __init__(self, state_dim, hidden_dim, action_dim):
        """初始化Q值网络的各个层

        Args:
            state_dim (int): 状态空间的维度
            hidden_dim (int): 隐藏层的神经元数量
            action_dim (int): 动作空间的维度
        """
        # 调用父类的初始化方法
        super(QValueNetContinuous, self).__init__()
        # 创建第一个全连接层，输入维度为状态维度加动作维度
        self.fc1 = torch.nn.Linear(state_dim + action_dim, hidden_dim)
        # 创建第二个隐藏层
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        # 创建输出层，输出单个Q值
        self.fc_out = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x, a):
        """前向传播函数，计算给定状态和动作的Q值

        Args:
            x (torch.Tensor): 状态张量
            a (torch.Tensor): 动作张量

        Returns:
            torch.Tensor: 对应的Q值
        """
        # 将状态和动作在最后一个维度上拼接
        cat = torch.cat([x, a], dim=1)
        # 通过第一个全连接层并应用ReLU激活函数
        x = F.relu(self.fc1(cat))
        # 通过第二个全连接层并应用ReLU激活函数
        x = F.relu(self.fc2(x))
        # 通过输出层得到最终的Q值
        return self.fc_out(x)



# 定义SAC算法类，用于处理连续动作空间
class SACContinuous:
    """处理连续动作的SAC算法

    Soft Actor-Critic (SAC) 是一种基于最大熵的强化学习算法，
    适用于连续动作空间。它同时优化策略和价值函数，并自动调节探索-利用平衡。

    Args:
        state_dim (int): 状态空间维度
        hidden_dim (int): 神经网络隐藏层维度
        action_dim (int): 动作空间维度
        action_bound (float): 动作边界
        actor_lr (float): 策略网络学习率
        critic_lr (float): 价值网络学习率
        alpha_lr (float): 温度参数学习率
        target_entropy (float): 目标熵值
        tau (float): 软更新参数
        gamma (float): 折扣因子
        device (torch.device): 计算设备
    """

    def __init__(self, state_dim, hidden_dim, action_dim, action_bound,
                 actor_lr, critic_lr, alpha_lr, target_entropy, tau, gamma,
                 device):
        """初始化SAC算法的所有组件

        Args:
            state_dim (int): 状态空间维度
            hidden_dim (int): 神经网络隐藏层维度
            action_dim (int): 动作空间维度
            action_bound (float): 动作边界
            actor_lr (float): 策略网络学习率
            critic_lr (float): 价值网络学习率
            alpha_lr (float): 温度参数学习率
            target_entropy (float): 目标熵值
            tau (float): 软更新参数
            gamma (float): 折扣因子
            device (torch.device): 计算设备
        """
        # 创建策略网络并移动到指定设备
        self.actor = PolicyNetContinuous(state_dim, hidden_dim, action_dim,
                                         action_bound).to(device)  # 策略网络
        # 创建第一个Q网络并移动到指定设备
        self.critic_1 = QValueNetContinuous(state_dim, hidden_dim,
                                            action_dim).to(device)  # 第一个Q网络
        # 创建第二个Q网络并移动到指定设备
        self.critic_2 = QValueNetContinuous(state_dim, hidden_dim,
                                            action_dim).to(device)  # 第二个Q网络
        # 创建第一个目标Q网络并移动到指定设备
        self.target_critic_1 = QValueNetContinuous(state_dim,
                                                   hidden_dim, action_dim).to(
                                                       device)  # 第一个目标Q网络
        # 创建第二个目标Q网络并移动到指定设备
        self.target_critic_2 = QValueNetContinuous(state_dim,
                                                   hidden_dim, action_dim).to(
                                                       device)  # 第二个目标Q网络
        # 将目标Q网络的参数初始化为与对应Q网络相同
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        # 将目标Q网络的参数初始化为与对应Q网络相同
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())
        # 创建策略网络的优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        # 创建第一个Q网络的优化器
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(),
                                                   lr=critic_lr)
        # 创建第二个Q网络的优化器
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(),
                                                   lr=critic_lr)
        # 初始化温度参数的对数值，使用对数形式可以保证alpha始终为正
        # ⚠️ 与标准公式差异：标准SAC论文建议初始α=1.0，这里设为0.01
        # 标准初始化：torch.tensor(np.log(1.0), dtype=torch.float)
        # 当前实现：较小的初始值可能影响初期探索行为
        self.log_alpha = torch.tensor(np.log(0.01), dtype=torch.float)
        # 设置log_alpha需要梯度，以便进行优化
        self.log_alpha.requires_grad = True  # 可以对alpha求梯度
        # 创建温度参数的优化器
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha],
                                                    lr=alpha_lr)
        # 保存目标熵值，用于自动调节温度参数
        self.target_entropy = target_entropy  # 目标熵的大小
        # 保存折扣因子
        self.gamma = gamma
        # 保存软更新参数
        self.tau = tau
        # 保存计算设备
        self.device = device

    def take_action(self, state):
        """根据当前状态选择动作

        Args:
            state: 当前环境状态

        Returns:
            list: 选择的动作列表
        """
        # 将状态转换为numpy数组再转换为tensor，避免性能警告
        state = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)
        # 通过策略网络获取动作，只取动作部分，忽略对数概率
        action = self.actor(state)[0]
        # 将tensor转换为标量并放入列表中返回
        return [action.item()]

    def calc_target(self, rewards, next_states, dones):
        """计算目标Q值

        Args:
            rewards (torch.Tensor): 奖励张量
            next_states (torch.Tensor): 下一状态张量
            dones (torch.Tensor): 结束标志张量

        Returns:
            torch.Tensor: 计算得到的目标Q值
        """
        # 使用策略网络获取下一状态的动作和对数概率
        next_actions, log_prob = self.actor(next_states)
        # 计算熵值（负对数概率）
        # ⚠️ 与标准公式差异：对于多维动作，标准熵应该是所有维度的负对数概率之和
        # 标准公式：entropy = -torch.sum(log_prob, dim=-1, keepdim=True)
        # 当前实现：适用于单维动作（Pendulum环境）
        entropy = -log_prob
        # 使用第一个目标Q网络计算Q值
        q1_value = self.target_critic_1(next_states, next_actions)
        # 使用第二个目标Q网络计算Q值
        q2_value = self.target_critic_2(next_states, next_actions)
        # 取两个Q值的最小值，并加上温度参数乘以熵值
        next_value = torch.min(q1_value,
                               q2_value) + self.log_alpha.exp() * entropy
        # 计算TD目标值：奖励 + 折扣因子 * 下一状态价值 * (1 - 结束标志)
        td_target = rewards + self.gamma * next_value * (1 - dones)
        # 返回计算得到的目标Q值
        return td_target

    def soft_update(self, net, target_net):
        """软更新目标网络参数

        Args:
            net (torch.nn.Module): 源网络
            target_net (torch.nn.Module): 目标网络
        """
        # 遍历源网络和目标网络的所有参数
        for param_target, param in zip(target_net.parameters(),
                                       net.parameters()):
            # 使用软更新公式更新目标网络参数：θ_target = (1-τ)*θ_target + τ*θ
            param_target.data.copy_(param_target.data * (1.0 - self.tau) +
                                    param.data * self.tau)

    def update(self, transition_dict):
        """更新SAC算法的所有网络参数

        Args:
            transition_dict (dict): 包含状态、动作、奖励等转换数据的字典
        """
        # 将状态数据转换为tensor并移动到指定设备
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)
        # 将动作数据转换为tensor，调整形状为列向量并移动到指定设备
        actions = torch.tensor(transition_dict['actions'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        # 将奖励数据转换为tensor，调整形状为列向量并移动到指定设备
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        # 将下一状态数据转换为tensor并移动到指定设备
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)
        # 将结束标志转换为tensor，调整形状为列向量并移动到指定设备
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)
        # 对倒立摆环境的奖励进行重塑，将奖励范围从[-8,0]映射到[0,1]以便训练
        # ⚠️ 与标准SAC差异：标准SAC算法不包含奖励重塑
        # 标准实现：直接使用原始奖励 rewards = rewards
        # 当前实现：为了训练稳定性添加的工程技巧，非算法本身的一部分
        rewards = (rewards + 8.0) / 8.0

        # 更新两个Q网络
        # 计算目标Q值
        td_target = self.calc_target(rewards, next_states, dones)
        # 计算第一个Q网络的损失，使用均方误差损失函数
        critic_1_loss = torch.mean(
            F.mse_loss(self.critic_1(states, actions), td_target.detach()))
        # 计算第二个Q网络的损失，使用均方误差损失函数
        critic_2_loss = torch.mean(
            F.mse_loss(self.critic_2(states, actions), td_target.detach()))
        # 清零第一个Q网络优化器的梯度
        self.critic_1_optimizer.zero_grad()
        # 反向传播计算第一个Q网络的梯度
        critic_1_loss.backward()
        # 更新第一个Q网络的参数
        self.critic_1_optimizer.step()
        # 清零第二个Q网络优化器的梯度
        self.critic_2_optimizer.zero_grad()
        # 反向传播计算第二个Q网络的梯度
        critic_2_loss.backward()
        # 更新第二个Q网络的参数
        self.critic_2_optimizer.step()

        # 更新策略网络
        # 使用策略网络重新采样动作和对数概率
        new_actions, log_prob = self.actor(states)
        # 计算熵值（负对数概率）
        # ⚠️ 与标准公式差异：对于多维动作，标准熵应该是所有维度的负对数概率之和
        # 标准公式：entropy = -torch.sum(log_prob, dim=-1, keepdim=True)
        # 当前实现：适用于单维动作（Pendulum环境）
        entropy = -log_prob
        # 使用第一个Q网络计算新动作的Q值
        q1_value = self.critic_1(states, new_actions)
        # 使用第二个Q网络计算新动作的Q值
        q2_value = self.critic_2(states, new_actions)
        # 计算策略损失：最大化Q值和熵值的加权和
        actor_loss = torch.mean(-self.log_alpha.exp() * entropy -
                                torch.min(q1_value, q2_value))
        # 清零策略网络优化器的梯度
        self.actor_optimizer.zero_grad()
        # 反向传播计算策略网络的梯度
        actor_loss.backward()
        # 更新策略网络的参数
        self.actor_optimizer.step()

        # 更新alpha值（温度参数）
        # 计算温度参数的损失：使熵值接近目标熵值
        alpha_loss = torch.mean(
            (entropy - self.target_entropy).detach() * self.log_alpha.exp())
        # 清零温度参数优化器的梯度
        self.log_alpha_optimizer.zero_grad()
        # 反向传播计算温度参数的梯度
        alpha_loss.backward()
        # 更新温度参数
        self.log_alpha_optimizer.step()

        # 软更新第一个目标Q网络的参数
        self.soft_update(self.critic_1, self.target_critic_1)
        # 软更新第二个目标Q网络的参数
        self.soft_update(self.critic_2, self.target_critic_2)


# 设置环境名称，使用v1版本替代已弃用的v0版本
env_name = 'Pendulum-v1'  # 使用v1版本替代已弃用的v0版本
# 创建强化学习环境
env = gym.make(env_name)
# 获取状态空间的维度
state_dim = env.observation_space.shape[0]  # 状态空间维度
# 获取动作空间的维度
action_dim = env.action_space.shape[0]  # 动作空间维度
# 获取动作的最大值（边界）
action_bound = env.action_space.high[0]  # 动作最大值
# 设置随机种子以确保结果可复现
random.seed(0)
# 设置numpy的随机种子
np.random.seed(0)
# gymnasium使用不同的随机种子设置方法
env.reset(seed=0)
# 设置PyTorch的随机种子
torch.manual_seed(0)

# SAC算法超参数设置
actor_lr = 3e-4      # 策略网络学习率
critic_lr = 3e-3     # 价值网络学习率
alpha_lr = 3e-4      # 温度参数学习率
num_episodes = 20    # 减少训练回合数用于快速测试
hidden_dim = 128     # 隐藏层维度
gamma = 0.99         # 折扣因子
tau = 0.005          # 软更新参数
buffer_size = 100000 # 经验回放池大小
minimal_size = 1000  # 开始训练的最小样本数
batch_size = 64      # 批次大小
target_entropy = -env.action_space.shape[0]  # 目标熵值
# ✅ 符合标准SAC：目标熵设为-动作维度，这是正确的标准实现
# 检查CUDA可用性，如果不可用则使用CPU
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"使用设备: {device}")

# 创建经验回放池
replay_buffer = rl_utils.ReplayBuffer(buffer_size)
print(f"经验回放池创建完成，容量: {buffer_size}")

# 创建SAC智能体
agent = SACContinuous(state_dim, hidden_dim, action_dim, action_bound,
                      actor_lr, critic_lr, alpha_lr, target_entropy, tau,
                      gamma, device)
print(f"SAC智能体创建完成")
print(f"状态维度: {state_dim}, 动作维度: {action_dim}, 动作边界: {action_bound}")

# 开始训练
print(f"开始训练，总回合数: {num_episodes}")
return_list = rl_utils.train_off_policy_agent(env, agent, num_episodes,
                                              replay_buffer, minimal_size,
                                              batch_size)

# 训练完成后的信息
print(f"训练完成！")
print(f"最后10个回合的平均回报: {np.mean(return_list[-10:]):.3f}")

# 绘制训练曲线
# 创建回合数列表，用于x轴
episodes_list = list(range(len(return_list)))
# 创建图形窗口，设置大小为10x6英寸
plt.figure(figsize=(10, 6))
# 绘制每个回合的奖励曲线
plt.plot(episodes_list, return_list, label='Episode Return')
# 绘制移动平均曲线，窗口大小为9
plt.plot(episodes_list, rl_utils.moving_average(return_list, 9), label='Moving Average')
# 设置x轴标签
plt.xlabel('Episodes')
# 设置y轴标签
plt.ylabel('Returns')
# 设置图形标题
plt.title('SAC Training on Pendulum-v1')
# 显示图例
plt.legend()
# 显示网格
plt.grid(True)
# 显示图形
plt.show()
# 打印提示信息
print("训练曲线已显示")