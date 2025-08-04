# 导入必要的库
import torch  # PyTorch深度学习框架
import numpy as np  # 数值计算库
try:
    import gymnasium as gym  # 新版本的gym库
except ImportError:
    import gym  # 如果没有gymnasium，使用旧版本gym
import matplotlib.pyplot as plt  # 绘图库
import torch.nn.functional as F  # PyTorch函数式API
import copy  # 深拷贝模块


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


def compute_advantage(gamma, lmbda, td_delta):
    """计算广义优势估计（GAE）

    这是一个计算广义优势估计（GAE）的函数，用于评估每个状态-动作对的优势值。
    GAE结合了n步优势估计的思想，通过指数加权平均的方式减少方差同时控制偏差。

    算法步骤：
    1. 将TD误差转换为NumPy数组
    2. 从后向前计算优势值，使用折扣因子和GAE参数
    3. 将结果转换回PyTorch张量

    Args:
        gamma (float): 折扣因子，用于计算未来奖励的现值，通常接近1（如0.99）
        lmbda (float): GAE平滑参数，控制偏差-方差权衡，取值范围[0,1]
        td_delta (torch.Tensor): 时序差分误差，即r + γV(s') - V(s)

    Returns:
        torch.Tensor: 计算得到的优势函数值，每个时间步一个值

    Note:
        当λ=0时，等价于单步TD误差
        当λ=1时，等价于蒙特卡洛估计
    """
    # 将TD误差转换为NumPy数组以便计算
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    # 从后向前计算GAE
    for delta in td_delta[::-1]:
        # 使用递推公式：A_t = δ_t + (γλ)A_(t+1)
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    # 反转列表使其按时间顺序排列
    advantage_list.reverse()
    # 转换回PyTorch张量并返回
    return torch.tensor(advantage_list, dtype=torch.float)


class PolicyNetContinuous(torch.nn.Module):
    """连续动作空间的策略网络

    该网络输出高斯分布的均值和标准差，用于连续动作空间的策略。

    Args:
        state_dim (int): 状态空间的维度
        hidden_dim (int): 隐藏层的维度
        action_dim (int): 动作空间的维度
    """
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNetContinuous, self).__init__()  # 调用父类构造函数
        # 第一层全连接层，从状态维度到隐藏维度
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        # 输出高斯分布均值的全连接层
        self.fc_mu = torch.nn.Linear(hidden_dim, action_dim)
        # 输出高斯分布标准差的全连接层
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


class TRPOContinuous:
    """处理连续动作空间的TRPO算法实现

    该类实现了适用于连续动作空间的TRPO算法，使用高斯分布来表示策略。

    Args:
        hidden_dim (int): 隐藏层的维度
        state_space (gym.Space): 环境的状态空间
        action_space (gym.Space): 环境的动作空间
        lmbda (float): GAE优势估计的λ参数
        kl_constraint (float): KL散度约束的阈值
        alpha (float): 线性搜索的步长参数
        critic_lr (float): 评论家网络的学习率
        gamma (float): 折扣因子
        device (torch.device): 计算设备（CPU/GPU）
    """
    def __init__(self, hidden_dim, state_space, action_space, lmbda,
                 kl_constraint, alpha, critic_lr, gamma, device):
        # 从环境空间获取维度信息
        state_dim = state_space.shape[0]  # 状态空间维度
        action_dim = action_space.shape[0]  # 动作空间维度（连续）

        # 创建演员（策略）网络并移至指定设备
        self.actor = PolicyNetContinuous(state_dim, hidden_dim,
                                         action_dim).to(device)
        # 创建评论家（价值）网络并移至指定设备
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        # 创建评论家网络的优化器
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)

        # 保存其他超参数
        self.gamma = gamma  # 折扣因子
        self.lmbda = lmbda  # GAE参数
        self.kl_constraint = kl_constraint  # KL散度约束
        self.alpha = alpha  # 线性搜索参数
        self.device = device  # 计算设备

    def take_action(self, state):
        """根据当前状态选择动作

        使用策略网络计算高斯分布的参数，然后从中采样一个连续动作。

        Args:
            state (np.ndarray): 当前环境状态

        Returns:
            np.ndarray: 选择的连续动作值
        """
        # 将状态转换为张量并移到指定设备
        state = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)
        # 使用策略网络计算高斯分布的均值和标准差
        mu, std = self.actor(state)
        # 创建正态分布
        action_dist = torch.distributions.Normal(mu, std)
        # 从分布中采样动作
        action = action_dist.sample()
        # 返回动作值（转换为numpy数组格式，适合gym环境）
        return action.detach().cpu().numpy().flatten()

    def hessian_matrix_vector_product(self,
                                      states,
                                      old_action_dists,
                                      vector,
                                      damping=0.1):
        """计算Hessian矩阵与向量的乘积（连续动作版本）

        这个函数实现了TRPO算法中的核心计算：Hessian-vector product。
        通过自动微分计算KL散度的二阶导数与给定向量的乘积，避免了
        直接计算和存储巨大的Hessian矩阵。

        数学原理：
        - H: KL散度D_KL(π_old || π_new)对策略参数θ的Hessian矩阵
        - v: 输入向量
        - 返回: H·v (Hessian矩阵与向量的乘积)

        Args:
            states (torch.Tensor): 状态批次，形状为[batch_size, state_dim]
            old_action_dists (torch.distributions.Normal): 旧策略的动作分布
            vector (torch.Tensor): 需要与Hessian矩阵相乘的向量
            damping (float): 阻尼系数，用于数值稳定性，默认0.1

        Returns:
            torch.Tensor: Hessian矩阵与输入向量的乘积，加上阻尼项
        """
        # 使用当前策略网络计算新的动作分布参数（均值和标准差）
        mu, std = self.actor(states)
        # 创建新的正态分布对象，用于计算KL散度
        new_action_dists = torch.distributions.Normal(mu, std)

        # 计算旧策略和新策略之间的平均KL散度
        # KL散度衡量两个概率分布之间的差异
        kl = torch.mean(
            torch.distributions.kl.kl_divergence(old_action_dists,
                                                 new_action_dists))

        # 计算KL散度对策略网络参数的一阶导数（梯度）
        # create_graph=True 允许对这个梯度再次求导
        kl_grad = torch.autograd.grad(kl,
                                      self.actor.parameters(),
                                      create_graph=True)

        # 将所有参数的梯度展平并连接成一个向量
        # 这样便于后续的向量运算
        kl_grad_vector = torch.cat([grad.view(-1) for grad in kl_grad])

        # 计算KL梯度向量与输入向量的点积
        # 这是计算Hessian-vector product的中间步骤
        kl_grad_vector_product = torch.dot(kl_grad_vector, vector)

        # 对点积结果再次求导，得到Hessian-vector product
        # 这利用了自动微分的链式法则
        grad2 = torch.autograd.grad(kl_grad_vector_product,
                                    self.actor.parameters())

        # 将二阶导数展平并连接成向量
        # contiguous()确保内存布局连续，提高计算效率
        grad2_vector = torch.cat(
            [grad.contiguous().view(-1) for grad in grad2])

        # 返回Hessian-vector product加上阻尼项
        # 阻尼项有助于数值稳定性和收敛性
        return grad2_vector + damping * vector

    def conjugate_gradient(self, grad, states, old_action_dists):
        """使用共轭梯度法求解线性方程组 Hx = g

        共轭梯度法是一种迭代算法，用于求解大型稀疏线性方程组。
        在TRPO中，我们需要求解 H·x = g，其中：
        - H: KL散度的Hessian矩阵（通过hessian_matrix_vector_product隐式计算）
        - g: 目标函数的梯度向量（输入参数grad）
        - x: 要求解的向量（策略更新方向）

        算法优势：
        1. 避免直接计算和存储Hessian矩阵（可能非常大）
        2. 只需要Hessian-vector product，计算效率高
        3. 数值稳定性好，适合大规模问题

        Args:
            grad (torch.Tensor): 目标函数的梯度向量 g
            states (torch.Tensor): 状态批次
            old_action_dists (torch.distributions.Normal): 旧策略的动作分布

        Returns:
            torch.Tensor: 方程 Hx = g 的解向量 x
        """
        # 初始化解向量x为零向量，与梯度向量同样大小
        x = torch.zeros_like(grad)

        # 初始化残差向量r为目标函数梯度g
        # 残差表示当前解与真实解的差距
        r = grad.clone()

        # 初始化搜索方向p为初始残差
        # 搜索方向指示下一步更新的方向
        p = grad.clone()

        # 计算初始残差的模长平方，用于收敛判断
        rdotr = torch.dot(r, r)

        # 共轭梯度主循环，最多迭代10次
        for i in range(10):
            # 计算Hessian矩阵与当前搜索方向的乘积
            # 这是共轭梯度法的核心步骤
            Hp = self.hessian_matrix_vector_product(states, old_action_dists, p)

            # 计算步长alpha，决定沿搜索方向移动多远
            # alpha = r^T·r / (p^T·H·p)
            alpha = rdotr / torch.dot(p, Hp)

            # 更新解向量：x = x + alpha * p
            x += alpha * p

            # 更新残差向量：r = r - alpha * H * p
            r -= alpha * Hp

            # 计算新的残差模长平方
            new_rdotr = torch.dot(r, r)

            # 收敛检查：如果残差足够小，提前退出
            if new_rdotr < 1e-10:
                break

            # 计算beta系数，用于更新搜索方向
            # beta = ||r_new||^2 / ||r_old||^2
            beta = new_rdotr / rdotr

            # 更新搜索方向：p = r + beta * p
            # 新的搜索方向结合了当前残差和之前的搜索方向
            p = r + beta * p

            # 更新残差模长平方，为下次迭代做准备
            rdotr = new_rdotr

        # 返回求解得到的向量x
        return x

    def compute_surrogate_obj(self, states, actions, advantage, old_log_probs,
                              actor):
        """计算替代目标函数（连续动作版本）

        替代目标函数是TRPO算法的核心组件，它近似表示策略改进的期望收益。
        通过最大化这个目标函数，我们可以找到更好的策略参数。

        数学公式：
        L(θ) = E[π_θ(a|s) / π_θ_old(a|s) * A(s,a)]
        其中：
        - π_θ(a|s): 新策略在状态s下选择动作a的概率
        - π_θ_old(a|s): 旧策略在状态s下选择动作a的概率
        - A(s,a): 优势函数，表示在状态s执行动作a的相对价值

        Args:
            states (torch.Tensor): 状态批次，形状为[batch_size, state_dim]
            actions (torch.Tensor): 动作批次，形状为[batch_size, action_dim]
            advantage (torch.Tensor): 优势函数值，形状为[batch_size, 1]
            old_log_probs (torch.Tensor): 旧策略下动作的对数概率
            actor (PolicyNetContinuous): 要评估的策略网络

        Returns:
            torch.Tensor: 替代目标函数值（标量）
        """
        # 使用给定的策略网络计算动作分布的参数（均值和标准差）
        mu, std = actor(states)

        # 创建正态分布对象，用于计算动作概率
        action_dists = torch.distributions.Normal(mu, std)

        # 计算新策略下动作的对数概率
        # 对于连续动作，这是概率密度函数的对数值
        log_probs = action_dists.log_prob(actions)

        # 计算重要性采样比率：π_new(a|s) / π_old(a|s)
        # 使用对数概率的差值再取指数，数值更稳定
        ratio = torch.exp(log_probs - old_log_probs)

        # 计算并返回替代目标函数的期望值
        # 这是重要性采样比率与优势函数的乘积的平均值
        return torch.mean(ratio * advantage)

    def line_search(self, states, actions, advantage, old_log_probs,
                    old_action_dists, max_vec):
        """执行线性搜索以确保策略更新满足约束条件

        线性搜索是TRPO算法的安全机制，确保策略更新既能改善性能，
        又能满足KL散度约束。通过逐步缩小步长，找到最优的更新幅度。

        搜索策略：
        1. 从最大步长开始尝试
        2. 如果不满足约束，将步长乘以alpha（通常0.5）
        3. 重复直到找到满足条件的步长或达到最大尝试次数

        约束条件：
        1. 新目标函数值 > 旧目标函数值（性能改善）
        2. KL散度 < KL约束阈值（策略变化不能太大）

        Args:
            states (torch.Tensor): 状态批次
            actions (torch.Tensor): 动作批次
            advantage (torch.Tensor): 优势函数值
            old_log_probs (torch.Tensor): 旧策略下动作的对数概率
            old_action_dists (torch.distributions.Normal): 旧策略的动作分布
            max_vec (torch.Tensor): 最大更新向量（由共轭梯度法计算得出）

        Returns:
            torch.Tensor: 最终选择的参数向量
        """
        # 将当前策略网络的所有参数展平为一个向量
        # 这样便于进行向量运算和参数更新
        old_para = torch.nn.utils.convert_parameters.parameters_to_vector(
            self.actor.parameters())

        # 计算当前策略的目标函数值作为基准
        old_obj = self.compute_surrogate_obj(states, actions, advantage,
                                             old_log_probs, self.actor)

        # 线性搜索主循环，最多尝试15次不同的步长
        for i in range(15):
            # 计算当前尝试的步长系数
            # 第i次尝试的系数为alpha^i，步长逐渐减小
            coef = self.alpha**i

            # 计算新的参数向量：旧参数 + 步长 * 更新方向
            new_para = old_para + coef * max_vec

            # 创建策略网络的深拷贝，避免修改原网络
            new_actor = copy.deepcopy(self.actor)

            # 将新的参数向量设置到拷贝的网络中
            torch.nn.utils.convert_parameters.vector_to_parameters(
                new_para, new_actor.parameters())

            # 使用新策略计算动作分布参数
            mu, std = new_actor(states)
            # 创建新的动作分布对象
            new_action_dists = torch.distributions.Normal(mu, std)

            # 计算新旧策略之间的KL散度
            # 这是衡量策略变化幅度的关键指标
            kl_div = torch.mean(
                torch.distributions.kl.kl_divergence(old_action_dists,
                                                     new_action_dists))

            # 计算新策略的目标函数值
            new_obj = self.compute_surrogate_obj(states, actions, advantage,
                                                 old_log_probs, new_actor)

            # 检查是否满足TRPO的两个约束条件：
            # 1. 性能改善：新目标函数值 > 旧目标函数值
            # 2. 策略约束：KL散度 < 预设阈值
            if new_obj > old_obj and kl_div < self.kl_constraint:
                # 如果满足条件，返回这个参数向量
                return new_para

        # 如果所有尝试都不满足条件，返回原始参数（保守策略）
        # 这确保了算法的稳定性，避免有害的更新
        return old_para

    def policy_learn(self, states, actions, old_action_dists, old_log_probs,
                     advantage):
        """执行策略网络的学习更新（TRPO核心算法）

        这是TRPO算法的核心方法，实现了完整的策略更新流程：
        1. 计算目标函数梯度
        2. 使用共轭梯度法求解更新方向
        3. 计算最大允许步长
        4. 执行线性搜索找到最优步长
        5. 更新策略网络参数

        TRPO的核心思想：
        - 在信任域内（KL散度约束）最大化策略改进
        - 保证每次更新都是安全的（单调改进）
        - 避免策略崩溃，确保训练稳定性

        Args:
            states (torch.Tensor): 状态批次
            actions (torch.Tensor): 动作批次
            old_action_dists (torch.distributions.Normal): 旧策略的动作分布
            old_log_probs (torch.Tensor): 旧策略下动作的对数概率
            advantage (torch.Tensor): 优势函数值
        """
        # 步骤1: 计算当前策略的替代目标函数值
        # 这是我们要最大化的目标
        surrogate_obj = self.compute_surrogate_obj(states, actions, advantage,
                                                   old_log_probs, self.actor)

        # 步骤2: 计算目标函数对策略参数的梯度
        # 这告诉我们参数应该朝哪个方向变化以改善性能
        grads = torch.autograd.grad(surrogate_obj, self.actor.parameters())

        # 将所有参数的梯度展平并连接成一个向量
        # detach()断开计算图，避免后续计算影响梯度
        obj_grad = torch.cat([grad.view(-1) for grad in grads]).detach()

        # 步骤3: 使用共轭梯度法求解自然梯度方向
        # 解方程 H·x = g，其中H是KL散度的Hessian矩阵，g是目标函数梯度
        # 得到的x是考虑了策略空间几何结构的最优更新方向
        descent_direction = self.conjugate_gradient(obj_grad, states,
                                                    old_action_dists)

        # 步骤4: 计算Hessian矩阵与更新方向的乘积
        # 这用于计算最大允许步长
        Hd = self.hessian_matrix_vector_product(states, old_action_dists,
                                                descent_direction)

        # 步骤5: 计算最大步长系数
        # 基于KL散度约束计算理论上的最大步长
        # 公式: max_coef = sqrt(2δ / (d^T H d))，其中δ是KL约束
        max_coef = torch.sqrt(2 * self.kl_constraint /
                              (torch.dot(descent_direction, Hd) + 1e-8))

        # 步骤6: 执行线性搜索找到实际的最优步长
        # 在理论最大步长的基础上，通过回溯搜索找到既满足约束又改善性能的步长
        new_para = self.line_search(states, actions, advantage, old_log_probs,
                                    old_action_dists,
                                    descent_direction * max_coef)

        # 步骤7: 将找到的最优参数设置到策略网络中
        # 完成一次TRPO策略更新
        torch.nn.utils.convert_parameters.vector_to_parameters(
            new_para, self.actor.parameters())

    def update(self, transition_dict):
        """更新策略网络和价值网络

        这是TRPO算法的主要更新函数，负责：
        1. 处理经验数据
        2. 计算优势函数
        3. 更新价值网络（Critic）
        4. 更新策略网络（Actor）

        Args:
            transition_dict (dict): 包含一个episode的经验数据
                - 'states': 状态序列
                - 'actions': 动作序列
                - 'rewards': 奖励序列
                - 'next_states': 下一状态序列
                - 'dones': 终止标志序列
        """
        # 数据预处理：将numpy数组转换为PyTorch张量并移到指定设备
        # 状态数据：[episode_length, state_dim]
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)

        # 动作数据：reshape为[episode_length, 1]以匹配连续动作格式
        actions = torch.tensor(transition_dict['actions'],
                               dtype=torch.float).view(-1, 1).to(self.device)

        # 奖励数据：reshape为[episode_length, 1]
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)

        # 下一状态数据：[episode_length, state_dim]
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)

        # 终止标志：[episode_length, 1]，1表示episode结束
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)

        # 奖励标准化：将Pendulum环境的奖励从[-16.27, 0]映射到[-1, 1]
        # 这有助于训练稳定性和收敛速度
        rewards = (rewards + 8.0) / 8.0  # 对奖励进行修改,方便训练

        # 计算时序差分目标值：r + γ * V(s') * (1 - done)
        # 如果episode结束(done=1)，则不考虑下一状态的价值
        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)

        # 计算时序差分误差：TD_target - V(s)
        # 这是计算优势函数的基础
        td_delta = td_target - self.critic(states)

        # 使用GAE计算优势函数
        # 优势函数衡量在特定状态执行特定动作的相对价值
        advantage = compute_advantage(self.gamma, self.lmbda,
                                      td_delta.cpu()).to(self.device)

        # 计算旧策略的动作分布和对数概率
        # 使用detach()确保这些值不参与梯度计算
        mu, std = self.actor(states)
        old_action_dists = torch.distributions.Normal(mu.detach(),
                                                      std.detach())
        old_log_probs = old_action_dists.log_prob(actions)

        # 更新价值网络（Critic）
        # 使用均方误差损失函数训练价值网络预测TD目标
        critic_loss = torch.mean(
            F.mse_loss(self.critic(states), td_target.detach()))

        # 价值网络梯度更新
        self.critic_optimizer.zero_grad()  # 清零梯度
        critic_loss.backward()             # 反向传播计算梯度
        self.critic_optimizer.step()       # 更新参数

        # 更新策略网络（Actor）
        # 使用TRPO算法进行安全的策略改进
        self.policy_learn(states, actions, old_action_dists, old_log_probs,
                          advantage)


def train_on_policy_agent(env, agent, num_episodes):
    """训练在线策略智能体的函数

    这个函数实现了在线策略强化学习的标准训练流程。
    在线策略意味着智能体使用当前策略收集经验，然后立即用这些经验更新策略。

    训练流程：
    1. 重置环境获得初始状态
    2. 使用当前策略与环境交互收集一个完整episode的经验
    3. 使用收集的经验更新策略和价值函数
    4. 重复上述过程直到达到指定的episode数量

    Args:
        env: gym/gymnasium环境实例，提供状态、动作、奖励等接口
        agent: 强化学习智能体实例，实现take_action和update方法
        num_episodes (int): 总训练回合数，每个回合是一个完整的episode

    Returns:
        list: 每个回合的累积奖励列表，用于评估训练效果
    """
    # 导入进度条库，用于显示训练进度
    from tqdm import tqdm

    # 存储每个episode的累积奖励，用于监控训练效果
    return_list = []

    # 将总训练分为10个阶段，便于观察训练进度
    for i in range(10):
        # 创建进度条，显示当前阶段的训练进度
        with tqdm(total=int(num_episodes/10), desc=f'Iteration {i}') as pbar:
            # 在当前阶段内训练指定数量的episodes
            for i_episode in range(int(num_episodes/10)):
                # 初始化当前episode的累积奖励
                episode_return = 0

                # 初始化经验存储字典，用于存储一个episode的所有经验
                transition_dict = {
                    'states': [],      # 状态序列
                    'actions': [],     # 动作序列
                    'next_states': [], # 下一状态序列
                    'rewards': [],     # 奖励序列
                    'dones': []        # 终止标志序列
                }

                # 重置环境获得初始状态，兼容不同版本的gym/gymnasium
                try:
                    # 新版本gymnasium的重置方法，返回(observation, info)
                    state, _ = env.reset()
                except (TypeError, ValueError):
                    # 旧版本gym的重置方法，可能只返回observation
                    state = env.reset()
                    # 处理某些环境返回tuple的情况
                    if isinstance(state, tuple):
                        state = state[0]

                # 初始化episode终止标志
                done = False

                # 开始一个episode的交互循环
                while not done:
                    # 使用当前策略选择动作
                    action = agent.take_action(state)

                    # 在环境中执行动作，兼容不同版本的gym/gymnasium
                    step_result = env.step(action)

                    # 处理不同版本gym返回值的差异
                    if len(step_result) == 5:
                        # 新版本gymnasium返回5个值：(obs, reward, terminated, truncated, info)
                        next_state, reward, terminated, truncated, _ = step_result
                        # 合并两种终止状态：terminated(任务完成) 或 truncated(时间限制)
                        done = terminated or truncated
                    else:
                        # 旧版本gym返回4个值：(obs, reward, done, info)
                        next_state, reward, done, _ = step_result

                    # 存储当前步的经验数据
                    transition_dict['states'].append(state)           # 当前状态
                    transition_dict['actions'].append(action)         # 执行的动作
                    transition_dict['next_states'].append(next_state) # 下一状态
                    transition_dict['rewards'].append(reward)         # 获得的奖励
                    transition_dict['dones'].append(done)             # 是否终止

                    # 更新状态，准备下一步交互
                    state = next_state
                    # 累积奖励，用于评估episode表现
                    episode_return += reward

                # episode结束，记录累积奖励
                return_list.append(episode_return)

                # 使用收集的经验更新智能体（策略和价值函数）
                agent.update(transition_dict)

                # 每10个episodes更新一次进度条显示
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({
                        # 显示当前总episode数
                        'episode': f'{num_episodes/10 * i + i_episode+1}',
                        # 显示最近10个episodes的平均奖励
                        'return': f'{np.mean(return_list[-10:]):.3f}'
                    })

                # 更新进度条
                pbar.update(1)

    # 返回所有episodes的奖励列表
    return return_list


def moving_average(a, window_size):
    """计算数组的移动平均值，包括处理边界情况

    该函数使用滑动窗口计算数组的移动平均值。对于数组两端的元素，
    使用特殊的处理方法确保输出数组长度与输入相同。

    Args:
        a (np.array): 需要计算移动平均的输入数组
        window_size (int): 移动平均的窗口大小，必须为奇数

    Returns:
        np.array: 计算得到的移动平均数组，长度与输入数组相同

    Note:
        函数分三部分计算移动平均：
        1. 开始部分：使用累积和除以递增的索引
        2. 中间部分：标准的移动平均计算
        3. 结束部分：使用反向累积和除以递减的索引
    """
    # 在数组开头插入0并计算累积和，用于后续的差分计算
    cumulative_sum = np.cumsum(np.insert(a, 0, 0))

    # 计算中间部分的移动平均：使用累积和的差值除以窗口大小
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size

    # 生成用于计算开始部分的索引序列（1,3,5,...,window_size-2）
    r = np.arange(1, window_size-1, 2)

    # 计算开始部分的移动平均：使用累积和除以递增的索引
    begin = np.cumsum(a[:window_size-1])[::2] / r

    # 计算结束部分的移动平均：使用反向累积和除以递减的索引，然后反转结果
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]

    # 将开始、中间和结束部分拼接成完整的移动平均数组
    return np.concatenate((begin, middle, end))


# ==================== 主程序：TRPO算法在Pendulum环境中的应用 ====================

# 设置训练超参数
# 这些参数的选择对算法性能有重要影响，需要根据具体环境调优
num_episodes = 2000      # 训练回合数：总共训练2000个episodes
hidden_dim = 128         # 神经网络隐藏层维度：控制网络容量和表达能力
gamma = 0.9              # 折扣因子：控制对未来奖励的重视程度，越接近1越重视长期奖励
lmbda = 0.9              # GAE参数：控制优势估计的偏差-方差权衡，越接近1方差越大但偏差越小
critic_lr = 1e-2         # 评论家网络学习率：控制价值函数的学习速度
kl_constraint = 0.00005  # KL散度约束：限制策略更新幅度，确保训练稳定性
alpha = 0.5              # 线性搜索步长衰减因子：控制回溯搜索的激进程度

# 设置计算设备
# 优先使用GPU加速训练，如果没有GPU则使用CPU
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"使用设备: {device}")

# 创建并配置强化学习环境
env_name = 'Pendulum-v1'  # Pendulum环境：经典的连续控制任务
print(f"创建环境: {env_name}")
env = gym.make(env_name)  # 实例化环境

# 环境信息打印
print(f"状态空间维度: {env.observation_space.shape}")
print(f"动作空间维度: {env.action_space.shape}")
print(f"动作范围: [{env.action_space.low[0]:.2f}, {env.action_space.high[0]:.2f}]")

# 设置随机种子确保实验可重现
# 这对于科学研究和算法比较非常重要
torch.manual_seed(0)     # 设置PyTorch随机种子
np.random.seed(0)        # 设置NumPy随机种子

# 重置环境并设置种子，兼容不同版本的gym/gymnasium
try:
    # 新版本gymnasium的API
    state, _ = env.reset(seed=0)
    print("使用Gymnasium API")
except TypeError:
    # 旧版本gym的API
    env.seed(0)
    state = env.reset()
    if isinstance(state, tuple):
        state = state[0]
    print("使用Gym API")

# 创建TRPO智能体实例
print("创建TRPO智能体...")
agent = TRPOContinuous(
    hidden_dim=hidden_dim,                    # 神经网络隐藏层维度
    state_space=env.observation_space,        # 环境状态空间
    action_space=env.action_space,            # 环境动作空间
    lmbda=lmbda,                             # GAE参数
    kl_constraint=kl_constraint,             # KL散度约束
    alpha=alpha,                             # 线性搜索参数
    critic_lr=critic_lr,                     # 评论家学习率
    gamma=gamma,                             # 折扣因子
    device=device                            # 计算设备
)

print("开始训练...")
print(f"训练参数: episodes={num_episodes}, hidden_dim={hidden_dim}, gamma={gamma}")
print(f"TRPO参数: kl_constraint={kl_constraint}, alpha={alpha}, lambda={lmbda}")

# 执行训练过程
# 这是整个程序的核心，智能体将通过与环境交互学习最优策略
return_list = train_on_policy_agent(env, agent, num_episodes)

# ==================== 训练结果分析和可视化 ====================

print("训练完成！开始分析结果...")

# 打印训练统计信息
print(f"总训练episodes: {len(return_list)}")
print(f"初始性能: {return_list[0]:.2f}")
print(f"最终性能: {return_list[-1]:.2f}")
print(f"最佳性能: {max(return_list):.2f}")
print(f"平均性能: {np.mean(return_list):.2f}")
print(f"最后100episodes平均性能: {np.mean(return_list[-100:]):.2f}")

# 创建用于绘图的回合数列表，长度与return_list相同
episodes_list = list(range(len(return_list)))

# 绘制原始训练结果图表
print("绘制原始训练曲线...")
plt.figure(figsize=(12, 5))

# 第一个子图：原始奖励曲线
plt.subplot(1, 2, 1)
plt.plot(episodes_list, return_list, alpha=0.7, color='blue')  # 绘制回合数-累积奖励曲线
plt.xlabel('Episodes')                                         # 设置x轴标签为回合数
plt.ylabel('Returns')                                          # 设置y轴标签为累积奖励
plt.title('TRPO Training Progress (Raw)')                      # 设置图表标题
plt.grid(True, alpha=0.3)                                     # 添加网格线

# 计算并绘制移动平均后的训练结果
print("计算移动平均并绘制平滑曲线...")
mv_return = moving_average(return_list, 9)  # 使用窗口大小为9计算移动平均

# 第二个子图：移动平均奖励曲线
plt.subplot(1, 2, 2)
plt.plot(episodes_list, mv_return, color='red', linewidth=2)   # 绘制回合数-移动平均累积奖励曲线
plt.xlabel('Episodes')                                         # 设置x轴标签为回合数
plt.ylabel('Returns')                                          # 设置y轴标签为累积奖励
plt.title('TRPO Training Progress (Smoothed)')                 # 设置图表标题
plt.grid(True, alpha=0.3)                                     # 添加网格线

# 调整子图间距并显示
plt.tight_layout()
plt.show()

# 绘制学习曲线分析图
print("绘制详细学习曲线分析...")
plt.figure(figsize=(15, 10))

# 子图1：完整训练曲线
plt.subplot(2, 2, 1)
plt.plot(episodes_list, return_list, alpha=0.5, label='Raw Returns')
plt.plot(episodes_list, mv_return, linewidth=2, label='Moving Average')
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('Complete Training Curve')
plt.legend()
plt.grid(True, alpha=0.3)

# 子图2：训练前期（前500episodes）
plt.subplot(2, 2, 2)
early_episodes = min(500, len(return_list))
plt.plot(episodes_list[:early_episodes], return_list[:early_episodes], alpha=0.7)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('Early Training (First 500 Episodes)')
plt.grid(True, alpha=0.3)

# 子图3：训练后期（后500episodes）
plt.subplot(2, 2, 3)
late_start = max(0, len(return_list) - 500)
plt.plot(episodes_list[late_start:], return_list[late_start:], alpha=0.7, color='green')
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('Late Training (Last 500 Episodes)')
plt.grid(True, alpha=0.3)

# 子图4：性能分布直方图
plt.subplot(2, 2, 4)
plt.hist(return_list, bins=50, alpha=0.7, color='purple', edgecolor='black')
plt.xlabel('Returns')
plt.ylabel('Frequency')
plt.title('Return Distribution')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("结果分析完成！")

# 关闭gym环境，释放相关资源
print("清理资源...")
env.close()
print("程序执行完毕！")