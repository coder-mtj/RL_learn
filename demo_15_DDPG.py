import random  # 用于随机数生成
import gymnasium as gym  # Gymnasium环境库（gym的维护版本）
import numpy as np  # 数值计算库
from tqdm import tqdm  # 进度条显示库
import torch  # PyTorch深度学习框架
from torch import nn  # 神经网络模块
import torch.nn.functional as F  # 神经网络函数库
import matplotlib.pyplot as plt  # 绘图库
import rl_utils  # 自定义的强化学习工具模块


class PolicyNet(torch.nn.Module):
    """策略网络（Actor网络）

    用于DDPG算法中的策略网络，输出连续动作值。
    使用tanh激活函数确保输出在[-1,1]范围内，然后乘以动作边界。

    Args:
        state_dim: int, 状态空间维度
        hidden_dim: int, 隐藏层维度
        action_dim: int, 动作空间维度
        action_bound: float, 动作的最大绝对值
    """

    def __init__(self, state_dim, hidden_dim, action_dim, action_bound):
        """初始化策略网络"""
        super(PolicyNet, self).__init__()
        # 第一个全连接层：状态维度 -> 隐藏层维度
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        # 第二个全连接层：隐藏层维度 -> 动作维度
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)
        # 动作边界，用于将tanh输出缩放到实际动作范围
        self.action_bound = action_bound

        # 权重初始化，有助于GPU训练稳定性
        self._init_weights()

    def _init_weights(self):
        """初始化网络权重，提高GPU训练稳定性"""
        # 使用Xavier初始化
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        # 偏置初始化为0
        torch.nn.init.zeros_(self.fc1.bias)
        torch.nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        """前向传播

        Args:
            x: torch.Tensor, 输入状态

        Returns:
            torch.Tensor, 输出动作，范围在[-action_bound, action_bound]
        """
        # 第一层：ReLU激活
        x = F.relu(self.fc1(x))
        # 第二层：tanh激活并缩放到动作边界
        # 这里首先使用tanh激活函数映射到(-1, 1)然后再*bound映射到对应的动作范围内
        return torch.tanh(self.fc2(x)) * self.action_bound


class QValueNet(torch.nn.Module):
    """Q值网络（Critic网络）

    用于DDPG算法中的价值网络，评估状态-动作对的Q值。
    网络输入为状态和动作的拼接，输出单个Q值。

    Args:
        state_dim: int, 状态空间维度
        hidden_dim: int, 隐藏层维度
        action_dim: int, 动作空间维度
    """

    def __init__(self, state_dim, hidden_dim, action_dim):
        """初始化Q值网络"""
        super(QValueNet, self).__init__()
        # 第一个全连接层：(状态维度+动作维度) -> 隐藏层维度
        self.fc1 = torch.nn.Linear(state_dim + action_dim, hidden_dim)
        # 第二个全连接层：隐藏层维度 -> 隐藏层维度
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        # 输出层：隐藏层维度 -> 1（Q值）
        self.fc_out = torch.nn.Linear(hidden_dim, 1)

        # 权重初始化，有助于GPU训练稳定性
        self._init_weights()

    def _init_weights(self):
        """初始化网络权重，提高GPU训练稳定性"""
        # 使用Xavier初始化
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.xavier_uniform_(self.fc_out.weight)
        # 偏置初始化为0
        torch.nn.init.zeros_(self.fc1.bias)
        torch.nn.init.zeros_(self.fc2.bias)
        torch.nn.init.zeros_(self.fc_out.bias)

    def forward(self, s, a):
        """前向传播

        Args:
            s: torch.Tensor, 输入状态
            a: torch.Tensor, 输入动作

        Returns:
            torch.Tensor, 输出Q值
        """
        # 将状态和动作在最后一个维度上拼接
        cat = torch.cat([s, a], dim=1)
        # 第一层：ReLU激活
        x = F.relu(self.fc1(cat))
        # 第二层：ReLU激活
        x = F.relu(self.fc2(x))
        # 输出层：直接输出Q值
        return self.fc_out(x)


class DDPG:
    """DDPG（Deep Deterministic Policy Gradient）算法实现类

    DDPG是一种基于Actor-Critic架构的深度强化学习算法，
    专门用于解决连续动作空间的强化学习问题。
    """

    def __init__(self, state_dim, hidden_dim, action_dim, action_bound, sigma, actor_lr, critic_lr, tau, gamma, device):
        """初始化DDPG算法的所有组件

        Args:
            state_dim: int, 状态空间维度
            hidden_dim: int, 神经网络隐藏层维度
            action_dim: int, 动作空间维度
            action_bound: float, 动作的最大绝对值边界
            sigma: float, 探索噪声的标准差
            actor_lr: float, Actor网络的学习率
            critic_lr: float, Critic网络的学习率
            tau: float, 目标网络软更新参数
            gamma: float, 折扣因子
            device: torch.device, 计算设备（CPU或GPU）
        """
        # 创建主Actor网络（策略网络）并移动到指定设备
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim, action_bound).to(device)
        # 创建主Critic网络（价值网络）并移动到指定设备
        self.critic = QValueNet(state_dim, hidden_dim, action_dim).to(device)
        # 创建目标Actor网络并移动到指定设备
        self.target_actor = PolicyNet(state_dim, hidden_dim, action_dim, action_bound).to(device)
        # 创建目标Critic网络并移动到指定设备
        self.target_critic = QValueNet(state_dim, hidden_dim, action_dim).to(device)

        # 将主Critic网络的参数复制到目标Critic网络
        self.target_critic.load_state_dict(self.critic.state_dict())
        # 将主Actor网络的参数复制到目标Actor网络
        self.target_actor.load_state_dict(self.actor.state_dict())

        # 创建Actor网络的优化器（使用Adam优化算法）
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        # 创建Critic网络的优化器（使用Adam优化算法）
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        # 保存折扣因子，用于计算未来奖励的现值
        self.gamma = gamma
        # 保存高斯噪声的标准差，用于动作探索（均值为0）
        self.sigma = sigma
        # 保存目标网络软更新参数，控制目标网络更新速度
        self.tau = tau
        # 保存动作空间维度
        self.action_dim = action_dim
        # 保存动作边界值
        self.action_bound = action_bound
        # 保存计算设备
        self.device = device

    def take_action(self, state):
        """根据当前状态选择动作（带探索噪声）

        这个方法实现了DDPG算法的动作选择策略，包括：
        1. 通过Actor网络生成确定性动作
        2. 添加高斯噪声进行探索
        3. 限制动作在有效范围内

        Args:
            state: np.array, 当前环境状态

        Returns:
            np.array, 选择的动作（已添加探索噪声并限制在有效范围内）
        """
        # 将numpy状态数组转换为PyTorch tensor格式
        # 添加batch维度（从shape [state_dim] 变为 [1, state_dim]）
        # 直接在GPU设备上创建tensor，避免后续的设备转移开销
        state = torch.tensor(np.array([state]), dtype=torch.float, device=self.device)

        # 使用torch.no_grad()上下文管理器进行推理
        # 这样可以禁用梯度计算，节省内存并提高推理速度
        with torch.no_grad():
            # 通过Actor网络前向传播获取确定性动作
            action = self.actor(state)

        # 将GPU上的tensor转移到CPU并转换为numpy数组
        # flatten()将shape从[1, action_dim]变为[action_dim]
        action = action.cpu().numpy().flatten()

        # 添加高斯噪声进行探索
        # np.random.randn生成标准正态分布随机数，乘以sigma调整噪声强度
        # 在CPU上进行噪声计算，避免GPU-CPU频繁数据传输
        action = action + self.sigma * np.random.randn(self.action_dim)

        # 使用np.clip将动作限制在环境允许的范围内
        # 确保动作在[-action_bound, action_bound]区间内
        action = np.clip(action, -self.action_bound, self.action_bound)

        # 返回最终的动作
        return action

    def soft_update(self, net, target_net):
        """软更新目标网络参数

        DDPG算法使用软更新策略来稳定训练过程。与硬更新（直接复制）不同，
        软更新通过线性插值的方式逐渐更新目标网络参数，公式为：
        θ_target = τ * θ_current + (1-τ) * θ_target
        其中τ是一个很小的值（如0.005），确保目标网络缓慢跟踪主网络。

        Args:
            net: 主网络（Actor或Critic）
            target_net: 对应的目标网络（Target Actor或Target Critic）
        """
        # 使用zip函数同时遍历目标网络和主网络的所有参数
        # target_net.parameters()返回目标网络的所有可训练参数
        # net.parameters()返回主网络的所有可训练参数
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            # 执行软更新公式：θ_target = (1-τ) * θ_target + τ * θ_current
            # param_target.data获取目标网络参数的数据部分
            # param.data获取主网络参数的数据部分
            # copy_()方法就地更新参数，不创建新的tensor
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

    def update(self, transition_dict):
        """更新DDPG算法的网络参数

        这是DDPG算法的核心更新函数，包含以下步骤：
        1. 准备训练数据（转换为GPU tensor）
        2. 更新Critic网络（最小化TD误差）
        3. 更新Actor网络（最大化Q值）
        4. 软更新目标网络

        Args:
            transition_dict: dict, 包含经验回放数据的字典，包含：
                - 'states': 当前状态列表
                - 'actions': 执行的动作列表
                - 'rewards': 获得的奖励列表
                - 'next_states': 下一状态列表
                - 'dones': 回合结束标志列表
        """
        # 将状态数据转换为PyTorch tensor并直接在GPU上创建
        # 先转换为numpy数组再创建tensor，避免性能警告
        # dtype=torch.float确保数据类型为32位浮点数
        # device=self.device确保tensor在正确的设备（GPU）上创建
        states = torch.tensor(np.array(transition_dict['states']), dtype=torch.float, device=self.device)

        # 将动作数据转换为GPU tensor并调整形状
        # 先转换为numpy数组，然后创建tensor，最后调整形状
        # view(-1, 1)将一维动作数组重塑为列向量形状[batch_size, 1]
        actions = torch.tensor(np.array(transition_dict['actions']), dtype=torch.float, device=self.device).view(-1, 1)

        # 将奖励数据转换为GPU tensor并调整形状为列向量
        # 先转换为numpy数组以提高性能
        rewards = torch.tensor(np.array(transition_dict['rewards']), dtype=torch.float, device=self.device).view(-1, 1)

        # 将下一状态数据转换为GPU tensor
        # 先转换为numpy数组再创建tensor
        next_states = torch.tensor(np.array(transition_dict['next_states']), dtype=torch.float, device=self.device)

        # 将回合结束标志转换为GPU tensor并调整形状为列向量
        # done=True表示回合结束，done=False表示回合继续
        # 先转换为numpy数组以避免性能警告
        dones = torch.tensor(np.array(transition_dict['dones']), dtype=torch.float, device=self.device).view(-1, 1)

        # ==================== 更新Critic网络 ====================
        # 第一步：使用目标网络计算下一状态的Q值
        # target_actor(next_states)：目标Actor网络根据下一状态生成动作
        # target_critic(next_states, ...)：目标Critic网络评估(下一状态,目标动作)的Q值
        next_q_values = self.target_critic(next_states, self.target_actor(next_states))

        # 第二步：计算TD目标值（Temporal Difference Target）
        # 公式：Q_target = r + γ * Q'(s', μ'(s')) * (1 - done)
        # rewards：即时奖励
        # self.gamma：折扣因子，控制未来奖励的重要性
        # (1 - dones)：如果回合结束(done=1)，则未来奖励为0
        q_targets = rewards + self.gamma * next_q_values * (1 - dones)

        # 第三步：计算Critic网络的损失函数
        # self.critic(states, actions)：当前Critic网络预测的Q值
        # F.mse_loss：计算预测Q值与目标Q值之间的均方误差
        # torch.mean：对批次中所有样本的损失取平均
        critic_loss = torch.mean(F.mse_loss(self.critic(states, actions), q_targets))

        # 第四步：反向传播更新Critic网络参数
        self.critic_optimizer.zero_grad()  # 清零之前累积的梯度
        critic_loss.backward()             # 计算损失函数关于参数的梯度
        self.critic_optimizer.step()       # 使用梯度更新Critic网络参数

        # ==================== 更新Actor网络 ====================
        # 第一步：计算Actor网络的损失函数
        # self.actor(states)：Actor网络根据状态生成动作
        # self.critic(states, ...)：Critic网络评估(状态,Actor动作)的Q值
        # 负号表示最大化Q值（梯度上升），等价于最小化-Q值（梯度下降）
        actor_loss = -torch.mean(self.critic(states, self.actor(states)))

        # 第二步：反向传播更新Actor网络参数
        self.actor_optimizer.zero_grad()   # 清零之前累积的梯度
        actor_loss.backward()              # 计算损失函数关于参数的梯度
        self.actor_optimizer.step()        # 使用梯度更新Actor网络参数

        # ==================== 软更新目标网络 ====================
        # 使用软更新策略缓慢更新目标网络，提高训练稳定性
        self.soft_update(self.actor, self.target_actor)    # 软更新目标Actor网络
        self.soft_update(self.critic, self.target_critic)  # 软更新目标Critic网络

# ==================== DDPG算法超参数设置 ====================
# Actor网络（策略网络）的学习率
# 较小的学习率有助于策略的稳定学习，避免策略震荡
actor_lr = 3e-4

# Critic网络（价值网络）的学习率
# 通常设置比Actor学习率大一个数量级，因为价值函数学习相对容易
critic_lr = 3e-3

# 总的训练回合数
# 每个回合智能体与环境交互直到回合结束
num_episodes = 200

# 神经网络隐藏层的维度（神经元数量）
# 控制网络的表达能力，过大可能过拟合，过小可能欠拟合
hidden_dim = 64

# 折扣因子γ，用于计算未来奖励的现值
# 接近1表示更重视长期奖励，接近0表示更重视即时奖励
gamma = 0.98

# 目标网络软更新参数τ
# 控制目标网络向主网络靠近的速度，较小值确保训练稳定
tau = 0.005

# 经验回放池的最大容量
# 存储历史经验用于训练，打破数据相关性
buffer_size = 10000

# 开始训练前回放池需要积累的最小样本数
# 确保有足够的经验数据进行有效的批次采样
minimal_size = 1000

# 每次训练时从回放池采样的批次大小
# 影响训练稳定性和计算效率的平衡
batch_size = 64

# 探索噪声的标准差σ
# 添加到动作上的高斯噪声，用于探索环境
sigma = 0.01
# ==================== 计算设备选择：严格使用CUDA ====================
# 检查CUDA是否可用，如果不可用则提供详细的安装指导
if not torch.cuda.is_available():
    # 打印错误信息和安装指导
    print("=" * 60)
    print("❌ CUDA不可用！")
    # 显示当前PyTorch版本信息
    print(f"当前PyTorch版本: {torch.__version__}")
    print("\n要使用CUDA进行训练，请按以下步骤操作：")

    # 第一步：检查硬件支持
    print("\n1. 检查您的GPU是否支持CUDA：")
    print("   - 运行 'nvidia-smi' 命令查看GPU信息")

    # 第二步：安装支持CUDA的PyTorch
    print("\n2. 安装支持CUDA的PyTorch版本：")
    print("   - 访问 https://pytorch.org/get-started/locally/")
    print("   - 选择您的CUDA版本（如CUDA 11.8或12.1）")
    print("   - 运行相应的安装命令，例如：")
    print("     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")

    # 第三步：重新运行
    print("\n3. 重新运行此脚本")
    print("=" * 60)

    # 抛出运行时错误，终止程序执行
    raise RuntimeError("CUDA不可用！请按上述步骤安装支持CUDA的PyTorch版本。")

# 创建CUDA设备对象，所有tensor和模型都将在此设备上运行
device = torch.device("cuda")

# 显示GPU设备信息，确认使用的硬件
# get_device_name(0)获取第0号GPU的名称
print(f"✅ 使用GPU设备: {torch.cuda.get_device_name(0)}")

# 显示CUDA版本信息
print(f"🔧 CUDA版本: {torch.version.cuda}")

# 显示GPU的总内存容量
# get_device_properties(0).total_memory返回字节数，除以1024^3转换为GB
print(f"💾 可用GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# ==================== 环境设置和初始化 ====================
# 指定要使用的强化学习环境名称
# Pendulum-v1是一个经典的连续控制任务：倒立摆平衡问题
env_name = 'Pendulum-v1'

# 使用gymnasium库创建环境实例
# 这将初始化倒立摆环境，包括物理模拟、渲染等功能
env = gym.make(env_name)

# ==================== 设置随机种子确保实验可重复 ====================
# 设置Python内置random模块的随机种子
# 这影响Python标准库中的随机数生成
random.seed(0)

# 设置NumPy库的随机种子
# 这影响NumPy数组操作和科学计算中的随机数生成
np.random.seed(0)

# 设置PyTorch的随机种子
# 这影响神经网络权重初始化、dropout等操作的随机性
torch.manual_seed(0)

# 注意：新版本gymnasium环境不再支持env.seed()方法
# 环境的随机性现在通过其他方式控制

# ==================== 创建经验回放池 ====================
# 实例化经验回放缓冲区，用于存储和采样训练数据
# buffer_size参数控制回放池的最大容量
replay_buffer = rl_utils.ReplayBuffer(buffer_size)

# ==================== 获取环境信息 ====================
# 获取状态空间的维度
# 对于倒立摆环境，状态包括：cos(θ), sin(θ), 角速度，共3维
state_dim = env.observation_space.shape[0]

# 获取动作空间的维度
# 对于倒立摆环境，动作是施加的扭矩，为1维连续值
action_dim = env.action_space.shape[0]

# 获取动作的最大绝对值边界
# 这定义了智能体可以输出的动作的有效范围
action_bound = env.action_space.high[0]

# ==================== 创建DDPG智能体 ====================
# 实例化DDPG算法，传入所有必要的超参数
# 这将创建Actor网络、Critic网络、目标网络和优化器
agent = DDPG(state_dim, hidden_dim, action_dim, action_bound, sigma,
             actor_lr, critic_lr, tau, gamma, device)

# ==================== GPU优化设置 ====================
# 清理GPU显存缓存，释放之前可能占用的内存
# 这有助于确保有足够的显存用于训练
torch.cuda.empty_cache()

# 启用cuDNN基准模式，自动寻找最优的卷积算法
# 虽然本例中没有卷积层，但这是GPU优化的标准做法
torch.backends.cudnn.benchmark = True

# 允许非确定性操作以提高GPU性能
# 设为False可以获得更好的性能，但会牺牲一些可重现性
torch.backends.cudnn.deterministic = False

# 显示训练开始前的GPU内存使用情况
# memory_allocated()返回当前分配的GPU内存（字节），转换为MB显示
print(f"训练前GPU内存使用: {torch.cuda.memory_allocated(0) / 1024**2:.1f} MB")

# ==================== 开始训练过程 ====================
# 打印训练开始信息，包括关键参数
print(f"🚀 开始训练DDPG算法...")
print(f"🎯 环境: {env_name}")
print(f"📊 状态维度: {state_dim}, 动作维度: {action_dim}")
print(f"💻 计算设备: {device}")
print(f"📦 批次大小: {batch_size}")
print(f"🔄 总训练回合数: {num_episodes}")
print("-" * 50)

# 调用训练函数开始离线策略智能体的训练过程
# 这将运行指定数量的回合，每个回合包含环境交互和网络更新
return_list = rl_utils.train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimal_size, batch_size)

# ==================== 训练完成后的信息显示 ====================
# 打印训练完成信息
print(f"✅ 训练完成！")

# 计算并显示最后10个回合的平均回报，用于评估训练效果
# np.mean()计算平均值，[-10:]选择列表的最后10个元素
print(f"📈 最后10个回合的平均回报: {np.mean(return_list[-10:]):.3f}")

# 显示训练结束后的GPU内存使用情况
print(f"💾 训练后GPU内存使用: {torch.cuda.memory_allocated(0) / 1024**2:.1f} MB")

# 显示训练过程中GPU内存的峰值使用量
# max_memory_allocated()返回自程序开始以来的最大内存分配量
print(f"🔝 GPU内存峰值使用: {torch.cuda.max_memory_allocated(0) / 1024**2:.1f} MB")

# 最后清理GPU缓存，释放不再需要的内存
torch.cuda.empty_cache()
print("🧹 GPU缓存已清理")