# 导入必要的库
import torch  # PyTorch深度学习框架
import numpy as np  # 数值计算库
import gymnasium as gym  # 统一使用gymnasium库
import matplotlib.pyplot as plt  # 绘图库
import torch.nn.functional as F  # PyTorch函数式API
import copy  # 深拷贝模块


class PolicyNet(torch.nn.Module):
    """策略网络(Actor)，用于生成动作的概率分布

    该网络将状态映射为动作概率分布，使用两层全连接神经网络实现。

    Args:
        state_dim (int): 状态空间的维度
        hidden_dim (int): 隐藏层的维度
        action_dim (int): 动作空间的维度
    """
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()  # 调用父类构造函数
        # 第一层全连接层，从状态维度到隐藏维度
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        # 第二层全连接层，从隐藏维度到动作维度
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        """前向传播函数

        Args:
            x (torch.Tensor): 输入状态张量

        Returns:
            torch.Tensor: 动作概率分布
        """
        # 使用ReLU激活函数处理第一层的输出
        x = F.relu(self.fc1(x))
        # 使用Softmax函数将输出转换为概率分布
        return F.softmax(self.fc2(x), dim=1)


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


class TRPO:
    """TRPO (Trust Region Policy Optimization) 算法的实现

    TRPO是一种基于置信域的策略优化算法，通过限制策略更新的KL散度来保证性能单调改善。

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
        action_dim = action_space.n  # 动作空间维度（离散）

        # 创建演员（策略）网络并移至指定设备
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
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

        使用策略网络计算动作概率分布，然后从中采样一个动作。

        Args:
            state (np.ndarray): 当前环境状态

        Returns:
            int: 选择的动作索引
        """
        # 将状态转换为张量并移到指定设备
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        # 使用策略网络计算动作概率分布
        probs = self.actor(state)
        # 创建类别分布
        action_dist = torch.distributions.Categorical(probs)
        # 从分布中采样动作
        action = action_dist.sample()
        # 返回动作索引
        return action.item()

    def hessian_matrix_vector_product(self, states, old_action_dists, vector):
        """计算Hessian矩阵与向量的乘积

        使用自动微分计算KL散度的二阶导数与向量的乘积。
        *** 这个函数隐式计算 H·v，其中H是Hessian矩阵，v是输入向量 ***

        Args:
            states (torch.Tensor): 状态批次
            old_action_dists (torch.distributions.Categorical): 旧策略的动作分布
            vector (torch.Tensor): 需要计算乘积的向量

        Returns:
            torch.Tensor: Hessian矩阵与输入向量的乘积 (H·v)
        """
        # 使用当前策略生成新的动作分布
        # self.actor(states)生成策略网络的输出（动作概率）
        # *** 这里涉及Hessian算法中的三个重要组件： ***
        # *** 1. g = ▽L(θ): 目标函数L(θ)对策略参数θ的一阶导数（梯度向量） ***
        # *** 2. H: KL散度D_KL(θ_old || θ)对参数θ的二阶导数（Hessian矩阵） ***
        # *** 3. x: 最终要求解的方向，满足Hx = g（Fisher信息矩阵与策略梯度的关系） ***
        new_action_dists = torch.distributions.Categorical(self.actor(states))
        # 计算新旧策略间的平均KL散度
        kl = torch.mean(torch.distributions.kl.kl_divergence(
            old_action_dists, new_action_dists))
        # 计算KL散度对策略参数的梯度
        kl_grad = torch.autograd.grad(kl,
                                    self.actor.parameters(),
                                    create_graph=True)
        # 将梯度展平并连接成一个向量
        kl_grad_vector = torch.cat([grad.view(-1) for grad in kl_grad])
        # 计算KL梯度向量与输入向量的点积
        kl_grad_vector_product = torch.dot(kl_grad_vector, vector)
        # 计算点积对策略参数的梯度（Hessian-vector product）
        grad2 = torch.autograd.grad(kl_grad_vector_product,
                                  self.actor.parameters())
        # 将结果展平并连接成一个向量
        grad2_vector = torch.cat([grad.view(-1) for grad in grad2])
        return grad2_vector

    def conjugate_gradient(self, grad, states, old_action_dists):
        """使用共轭梯度法求解线性方程组

        解决方程 Hx = g，其中H是KL散度的Hessian矩阵，g是目标函数的梯度。

        Args:
            grad (torch.Tensor): 目标函数关于策略参数的梯度 (g)
            states (torch.Tensor): 状态批次
            old_action_dists (torch.distributions.Categorical): 旧策略的动作分布

        Returns:
            torch.Tensor: 方程的解 (x)

        Note:
            这里使用共轭梯度法避免直接计算和存储Hessian矩阵
            *** 求解 Hx = g 方程，其中: ***
            *** H: KL散度的Hessian矩阵 (隐式通过hessian_matrix_vector_product计算) ***
            *** g: 目标函数的梯度向量 (输入参数grad) ***
            *** x: 要求解的向量 (返回值) ***
        """
        # *** 初始化解向量x为0向量 (Hessian算法中的x) ***
        x = torch.zeros_like(grad)
        # 初始化残差r和搜索方向p
        # *** r: 残差向量，初始值为g (目标函数梯度) ***
        r = grad.clone()
        # *** p: 搜索方向向量 ***
        p = grad.clone()
        # 计算初始残差的模平方
        rdotr = torch.dot(r, r)

        # 共轭梯度主循环
        for i in range(10):
            # *** 计算Hp: Hessian矩阵H与搜索方向p的乘积 ***
            Hp = self.hessian_matrix_vector_product(states, old_action_dists, p)
            # 计算步长
            alpha = rdotr / torch.dot(p, Hp)
            # 更新解向量
            x += alpha * p
            # 更新残差
            r -= alpha * Hp
            # 计算新残差的模平方
            new_rdotr = torch.dot(r, r)
            # 收敛检查
            if new_rdotr < 1e-10:
                break
            # 计算beta系数
            beta = new_rdotr / rdotr
            # 更新搜索方向
            p = r + beta * p
            rdotr = new_rdotr
        return x

    def compute_surrogate_obj(self, states, actions, advantage, old_log_probs, actor):
        """计算替代目标函数

        计算策略更新的替代目标函数值，这是TRPO算法中的关键部分。

        Args:
            states (torch.Tensor): 状态批次
            actions (torch.Tensor): 动作批次
            advantage (torch.Tensor): 优势函数值
            old_log_probs (torch.Tensor): 旧策略下动作的对数概率
            actor (PolicyNet): 要评估的策略网络

        Returns:
            torch.Tensor: 替代目标函数值
        """
        # 计算新策略下动作的对数概率
        log_probs = torch.log(actor(states).gather(1, actions))
        # 计算概率比率
        ratio = torch.exp(log_probs - old_log_probs)
        # 计算并返回替代目标函数值（期望优势估计）
        return torch.mean(ratio * advantage)

    def line_search(self, states, actions, advantage, old_log_probs,
                   old_action_dists, max_vec):
        """执行线性搜索以确保策略更新是保守的

        在策略更新方向上执行线性搜索，确保新策略满足KL散度约束且性能有所提升。

        Args:
            states (torch.Tensor): 状态批次
            actions (torch.Tensor): 动作批次
            advantage (torch.Tensor): 优势函数值
            old_log_probs (torch.Tensor): 旧策略下动作的对数概率
            old_action_dists (torch.distributions.Categorical): 旧策略的动作分布
            max_vec (torch.Tensor): 最大更新向量

        Returns:
            torch.Tensor: 搜索到的最优参数向量
        """
        # 获取当前策略参数
        old_para = torch.nn.utils.convert_parameters.parameters_to_vector(
            self.actor.parameters())
        # 计算当前目标函数值
        old_obj = self.compute_surrogate_obj(states, actions, advantage,
                                           old_log_probs, self.actor)

        # 线性搜索循环
        for i in range(15):
            # 计算当前步长
            coef = self.alpha**i
            # 计算新的参数向量
            new_para = old_para + coef * max_vec
            # 创建演员网络的副本
            new_actor = copy.deepcopy(self.actor)
            # 更新副本的参数
            torch.nn.utils.convert_parameters.vector_to_parameters(
                new_para, new_actor.parameters())
            # 计算新策略的动作分布
            new_action_dists = torch.distributions.Categorical(
                new_actor(states))
            # 计算KL散度
            kl_div = torch.mean(
                torch.distributions.kl.kl_divergence(old_action_dists,
                                                   new_action_dists))
            # 计算新的目标函数值
            new_obj = self.compute_surrogate_obj(states, actions, advantage,
                                               old_log_probs, new_actor)

            # 如果满足条件则接受更新
            if new_obj > old_obj and kl_div < self.kl_constraint:
                return new_para
        # 如果没找到合适的更新，返回原参数
        return old_para

    def policy_learn(self, states, actions, old_action_dists, old_log_probs, advantage):
        """更新策略网络参数

        使用TRPO算法的核心步骤更新策略网络，包括：
        1. 计算替代目标函数
        2. 使用共轭梯度法计算更新方向
        3. 进行线性搜索确保更新满足约束

        Args:
            states (torch.Tensor): 状态批次
            actions (torch.Tensor): 动作批次
            old_action_dists (torch.distributions.Categorical): 旧策略的动作分布
            old_log_probs (torch.Tensor): 旧策略下动作的对数概率
            advantage (torch.Tensor): 优势函数值

        Note:
            这个方法实现了TRPO的核心思想：在保证性能提升的同时，
            限制新旧策略之间的KL散度，从而实现稳定的策略更新。
        """
        # 计算当前策略的替代目标函数
        surrogate_obj = self.compute_surrogate_obj(states, actions, advantage,
                                                 old_log_probs, self.actor)

        # 计算目标函数关于策略参数的梯度
        grads = torch.autograd.grad(surrogate_obj, self.actor.parameters())
        # 将所有梯度展平并连接成一个向量，同时分离计算图
        # *** g: 目标函数的梯度向量 (Hessian算法中的g) ***
        obj_grad = torch.cat([grad.view(-1) for grad in grads]).detach()

        # 使用共轭梯度法计算更新方向：解决 Hx = g，其中H是KL散度的Hessian矩阵
        # *** x: 求解方程Hx=g的解向量 (Hessian算法中的x) ***
        descent_direction = self.conjugate_gradient(obj_grad, states,
                                                  old_action_dists)

        # 计算Hessian矩阵与下降方向的乘积
        # *** H: KL散度的Hessian矩阵 (通过hessian_matrix_vector_product隐式计算Hx) ***
        Hd = self.hessian_matrix_vector_product(states, old_action_dists,
                                               descent_direction)

        # 计算最大步，确保满足KL散度约束
        max_coef = torch.sqrt(2 * self.kl_constraint /
                            (torch.dot(descent_direction, Hd) + 1e-8))

        # 执行线性搜索，找到最优的参数更新
        new_para = self.line_search(states, actions, advantage, old_log_probs,
                                  old_action_dists,
                                  descent_direction * max_coef)

        # 使用找到的最优参数更新策略网络
        torch.nn.utils.convert_parameters.vector_to_parameters(
            new_para, self.actor.parameters())

    def update(self, transition_dict):
        """更新策略和价值函数

        Args:
            transition_dict: dict, 包含训练所需的各种数据
        """
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

        # 计算时序差分目标和误差
        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        td_delta = td_target - self.critic(states)
        # 使用自定义的compute_advantage替代rl_utils中的函数
        advantage = compute_advantage(self.gamma, self.lmbda,
                                    td_delta.cpu()).to(self.device)
        old_log_probs = torch.log(self.actor(states).gather(1,
                                                            actions)).detach()
        old_action_dists = torch.distributions.Categorical(
            self.actor(states).detach())
        critic_loss = torch.mean(
            F.mse_loss(self.critic(states), td_target.detach()))

        # 更新评论家网络
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # 更新演员网络
        self.policy_learn(states, actions, old_action_dists, old_log_probs,
                          advantage)



# 设置训练参数
num_episodes = 500  # 训练回合数
hidden_dim = 128    # 隐藏层维度
gamma = 0.98        # 折扣因子
lmbda = 0.95       # GAE参数
critic_lr = 1e-2    # 评论家网络学习率
kl_constraint = 0.0005  # KL散度约束
alpha = 0.5        # 线性搜索参数

# 设置运行设备（如果有GPU则使用GPU，否则使用CPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 创建并设置环境
env_name = 'CartPole-v1'  # 使用v1版本替代v0
env = gym.make(env_name)  # 创建环境实例

# 设置随机种子以确保实验可重现
torch.manual_seed(0)  # 设置PyTorch的随机种子
state, _ = env.reset(seed=0)  # 重置环境并设置种子

# 创建TRPO智能体
agent = TRPO(hidden_dim, env.observation_space, env.action_space, lmbda,
             kl_constraint, alpha, critic_lr, gamma, device)

def train_on_policy_agent(env, agent, num_episodes):
    """训练在线策略智能体的函数

    Args:
        env: gym环境实例
        agent: 强化学习智能体实例
        num_episodes: int, 总训练回合数

    Returns:
        list: 每个回合的累积奖励列表
    """
    from tqdm import tqdm
    return_list = []

    for i in range(10):
        with tqdm(total=int(num_episodes/10), desc=f'Iteration {i}') as pbar:
            for i_episode in range(int(num_episodes/10)):
                episode_return = 0
                transition_dict = {
                    'states': [],
                    'actions': [],
                    'next_states': [],
                    'rewards': [],
                    'dones': []
                }

                state, _ = env.reset()
                done = False

                while not done:
                    action = agent.take_action(state)
                    # 执行动作，处理新版本gym返回的5个值
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated  # 合并两种终止状态

                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(done)

                    state = next_state
                    episode_return += reward

                return_list.append(episode_return)
                agent.update(transition_dict)

                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({
                        'episode': f'{num_episodes/10 * i + i_episode+1}',
                        'return': f'{np.mean(return_list[-10:]):.3f}'
                    })
                pbar.update(1)

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

    # 计算结束部分的移��平均：使用反向累积和除以递减的索引，然后反转结���
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]

    # 将开始、中间和结束部��拼接成完整的移动平均数组
    return np.concatenate((begin, middle, end))

# 使用train_on_policy_agent函数训练智能体，并获取每个回合的累积奖励列表
return_list = train_on_policy_agent(env, agent, num_episodes)

# 创建用于绘图的回合数列表，长度与return_list相同
episodes_list = list(range(len(return_list)))

# 绘制原始训练结果图表
plt.plot(episodes_list, return_list)  # 绘制回合数-累积奖励曲线
plt.xlabel('Episodes')  # 设置x轴标签为回合数
plt.ylabel('Returns')   # 设置y轴标签为累积奖励
plt.title('TRPO on {}'.format(env_name))  # 设置图表标题
plt.show()  # 显示图表

# 计算并绘制移动平均后的训练结果
mv_return = moving_average(return_list, 9)  # 使用窗口大小为9计算移动平均
plt.plot(episodes_list, mv_return)  # 绘制回合数-移动平均累积奖励曲���
plt.xlabel('Episodes')  # 设置x轴标签为回合数
plt.ylabel('Returns')   # 设置y轴标签为累积奖励
plt.title('TRPO on {}'.format(env_name))  # 设置图表标题
plt.show()  # 显示图表

# 关闭gym环境，释放相关资源
env.close()
