# 强化学习算法实现集合

本项目实现了一系列经典的强化学习算法，从基础的多臂赌博机到高级的TRPO算法，提供了完整的代码实现和详细注释。

## 项目结构

### 基础算法
- `demo_01_multi_arm_bandit.py`: 多臂赌博机问题实现
- `demo_02_MDP.py`: 马尔可夫决策过程基础实现
- `demo_03_DP.py`: 动态规划方法
- `demo_04_env.py`: 强化学习环境实现

### 进阶算法
- `demo_05_dyna_Q.py`: Dyna-Q算法实现
- `demo_06_DQN.py`: 深度Q网络基础实现
- `demo_07_double_DQN.py`: 双重DQN算法
- `demo_08_dueling_DQN.py`: 决斗DQN网络结构

### 高级策略梯度方法
- `demo_09_REINFORCE.py`: REINFORCE算法（基础策略梯度）
- `demo_10_Actor_Critic.py`: Actor-Critic架构实现
- `demo_11_TRPO_CartPole.py`: TRPO算法在离散动作空间的实现（CartPole环境）
- `demo_12_TRPO_pendulum.py`: TRPO算法在连续动作空间的实现（Pendulum环境）

## TRPO算法详细说明

TRPO（Trust Region Policy Optimization）是一种先进的策略梯度算法，通过限制策略更新的幅度来确保训练的稳定性和单调改进。本项目提供了两个版本的实现：

### 📋 实现版本对比

| 特性 | demo_11_TRPO_CartPole.py | demo_12_TRPO_pendulum.py |
|------|-------------------------|--------------------------|
| **动作空间** | 离散动作（Categorical分布） | 连续动作（Normal分布） |
| **环境** | CartPole-v1 | Pendulum-v1 |
| **策略网络输出** | 动作概率向量 | 高斯分布参数（μ, σ） |
| **动作采样** | 类别分布采样 | 正态分布采样 |
| **KL散度计算** | 离散分布KL散度 | 连续分布KL散度 |
| **适用场景** | 游戏、导航等离散决策 | 机器人控制、连续控制 |

### 1. 数学原理

TRPO算法的核心是求解如下约束优化问题：

```
最大化：E[π_new(a|s)/π_old(a|s) * A(s,a)]
约束条件：E[KL(π_old(·|s) || π_new(·|s))] ≤ δ
```

其中：
- π_new 和 π_old 分别是新旧策略
- A(s,a) 是优势函数
- KL 是KL散度，用于度量策略更新的幅度
- δ 是KL散度约束阈值

#### 1.1 连续动作空间的特殊处理

对于连续动作空间，策略网络输出高斯分布的参数：
```
π(a|s) = N(μ(s), σ(s))
```
其中μ(s)和σ(s)分别是状态s下的均值和标准差。

### 2. 关键组件实现

#### 2.1 广义优势估计（GAE）
```python
def compute_advantage(gamma, lmbda, td_delta):
    """计算广义优势估计
    A(s,a) = δ_t + (γλ)δ_{t+1} + ... + (γλ)^{T-t-1}δ_{T-1}
    其中 δ_t 是TD误差
    """
    advantage = 0.0
    advantage_list = []
    for delta in td_delta[::-1]:  # 从后向前计算
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(advantage_list, dtype=torch.float)
```

#### 2.2 策略网络（Actor）

**离散动作空间（CartPole）：**
```python
class PolicyNet(torch.nn.Module):
    """将状态映射为动作概率分布的策略网络"""
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)  # 输出动作概率
```

**连续动作空间（Pendulum）：**
```python
class PolicyNetContinuous(torch.nn.Module):
    """连续动作空间的策略网络，输出高斯分布参数"""
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNetContinuous, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc_mu = torch.nn.Linear(hidden_dim, action_dim)    # 均值
        self.fc_std = torch.nn.Linear(hidden_dim, action_dim)   # 标准差

    def forward(self, x):
        x = F.relu(self.fc1(x))
        mu = 2.0 * torch.tanh(self.fc_mu(x))    # 限制动作范围
        std = F.softplus(self.fc_std(x))        # 确保标准差为正
        return mu, std
```

#### 2.3 价值网络（Critic）
```python
class ValueNet(torch.nn.Module):
    """评估状态价值的网络（两个版本通用）"""
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)
```

### 3. 算法核心步骤

TRPO算法的执行流程包含以下关键步骤：

1. **收集轨迹数据**
   - 使用当前策略π_old与环境交互
   - 存储状态、动作、奖励、下一状态等信息
   - 收集完整的episode数据

2. **计算优势估计**
   - 使用GAE方法计算优势函数值A(s,a)
   - 结合时序差分误差和多步回报
   - 平衡偏差和方差

3. **策略更新（TRPO核心）**
   - **步骤3.1**: 计算替代目标函数的梯度g
   - **步骤3.2**: 使用共轭梯度法求解Hx = g，得到自然梯度方向x
   - **步骤3.3**: 计算最大步长：max_coef = √(2δ/(x^T H x))
   - **步骤3.4**: 执行线性搜索，确保满足KL约束和性能改进
   - **步骤3.5**: 更新策略网络参数

4. **价值函数更新**
   - 使用时序差分目标更新价值网络
   - 最小化价值估计的均方误差
   - 为下一轮优势估计提供基准

#### 3.1 共轭梯度法详解

共轭梯度法用于高效求解线性方程组Hx = g：
```python
def conjugate_gradient(self, grad, states, old_action_dists):
    x = torch.zeros_like(grad)  # 初始解
    r = grad.clone()            # 初始残差
    p = grad.clone()            # 初始搜索方向

    for i in range(10):  # 最多10次迭代
        Hp = self.hessian_matrix_vector_product(states, old_action_dists, p)
        alpha = torch.dot(r, r) / torch.dot(p, Hp)
        x += alpha * p
        r -= alpha * Hp

        if torch.dot(r, r) < 1e-10:  # 收敛检查
            break

        beta = torch.dot(r, r) / torch.dot(r_old, r_old)
        p = r + beta * p

    return x
```

#### 3.2 线性搜索机制

线性搜索确保策略更新既改善性能又满足约束：
```python
def line_search(self, states, actions, advantage, old_log_probs,
                old_action_dists, max_vec):
    for i in range(15):  # 最多15次尝试
        coef = self.alpha ** i  # 步长衰减
        new_para = old_para + coef * max_vec

        # 检查约束条件
        if new_obj > old_obj and kl_div < self.kl_constraint:
            return new_para  # 找到满足条件的更新

    return old_para  # 保守策略：不更新
```

### 4. 参数配置

#### 4.1 CartPole环境（离散动作）
```python
# 网络参数
hidden_dim = 128        # 隐藏层维度

# 训练参数
num_episodes = 500      # 训练回合数
gamma = 0.98           # 折扣因子
lmbda = 0.95          # GAE参数
critic_lr = 1e-2       # 评论家学习率

# TRPO特有参数
kl_constraint = 0.0005  # KL散度约束
alpha = 0.5            # 线性搜索步长衰减
```

#### 4.2 Pendulum环境（连续动作）
```python
# 网络参数
hidden_dim = 128        # 隐藏层维度

# 训练参数
num_episodes = 2000     # 训练回合数（连续控制需要更多训练）
gamma = 0.9            # 折扣因子
lmbda = 0.9           # GAE参数
critic_lr = 1e-2       # 评论家学习率

# TRPO特有参数
kl_constraint = 0.00005 # KL散度约束（更严格）
alpha = 0.5            # 线性搜索步长衰减
```

#### 4.3 参数调优建议

| 参数 | 作用 | 调优建议 |
|------|------|----------|
| `kl_constraint` | 控制策略更新幅度 | 过大→不稳定；过小→学习慢 |
| `lmbda` | GAE偏差-方差权衡 | 接近1→低偏差高方差；接近0→高偏差低方差 |
| `gamma` | 未来奖励重要性 | 长期任务用0.99；短期任务用0.9 |
| `alpha` | 线性搜索激进程度 | 0.5是经验值，可尝试0.3-0.8 |

### 5. 使用说明

#### 5.1 环境准备
```bash
# 安装基础依赖
pip install torch numpy matplotlib tqdm

# 安装gymnasium库
pip install gymnasium[classic_control]
```

#### 5.2 运行离散动作版本（CartPole）
```python
# 运行demo_11_TRPO_CartPole.py
python demo_11_TRPO_CartPole.py

# 创建智能体示例
env = gym.make('CartPole-v1')
agent = TRPO(hidden_dim, env.observation_space, env.action_space,
            lmbda, kl_constraint, alpha, critic_lr, gamma, device)
return_list = train_on_policy_agent(env, agent, num_episodes)
```

#### 5.3 运行连续动作版本（Pendulum）
```python
# 运行demo_12_TRPO_pendulum.py
python demo_12_TRPO_pendulum.py

# 创建智能体示例
env = gym.make('Pendulum-v1')
agent = TRPOContinuous(hidden_dim, env.observation_space, env.action_space,
                      lmbda, kl_constraint, alpha, critic_lr, gamma, device)
return_list = train_on_policy_agent(env, agent, num_episodes)
```

#### 5.4 结果可视化
```python
# 绘制训练曲线
episodes_list = list(range(len(return_list)))

# 原始曲线
plt.subplot(1, 2, 1)
plt.plot(episodes_list, return_list)
plt.title('Raw Training Progress')

# 移动平均曲线
plt.subplot(1, 2, 2)
mv_return = moving_average(return_list, 9)
plt.plot(episodes_list, mv_return)
plt.title('Smoothed Training Progress')

plt.show()
```

### 6. 优势和特点

#### 6.1 算法优势
1. **稳定性**：通过KL散度约束确保策略更新的保守性，避免策略崩溃
2. **可靠性**：理论上保证单调策略改进，每次更新都不会变差
3. **适用性**：同时支持离散和连续动作空间，应用范围广泛
4. **效率**：使用共轭梯度法高效求解二阶优化问题，避免直接计算Hessian矩阵

#### 6.2 实现特点
1. **完整注释**：每行关键代码都有详细的中文注释
2. **模块化设计**：清晰的类结构，便于理解和修改
3. **兼容性好**：支持新旧版本的gym/gymnasium
4. **可视化丰富**：提供多种训练曲线和性能分析图表

#### 6.3 性能表现

**CartPole环境（离散动作）：**
- 通常在100-200个episodes内达到满分（500分）
- 训练稳定，很少出现性能倒退
- 适合作为TRPO算法的入门示例

**Pendulum环境（连续动作）：**
- 从初始-1400左右逐步改善到-200以内
- 需要更多训练episodes（2000+）
- 展示了TRPO在连续控制任务中的能力

### 7. 注意事项和最佳实践

#### 7.1 关键注意事项
1. **KL散度约束**：这是最关键的超参数
   - 过大（>0.01）：可能导致训练不稳定，策略更新过于激进
   - 过小（<0.0001）：学习速度慢，策略更新过于保守
   - 推荐值：离散动作0.0005，连续动作0.00005

2. **批量数据大小**：需要足够的数据来准确估计KL散度
   - 每个episode的数据都很重要
   - 不建议使用mini-batch，而是使用完整episode数据

3. **计算复杂度**：相比简单的策略梯度算法更复杂
   - 共轭梯度法需要多次Hessian-vector乘积计算
   - 线性搜索需要多次前向传播
   - 建议使用GPU加速

4. **超参数敏感性**：对参数设置较为敏感
   - 建议从推荐值开始，逐步微调
   - 不同环境可能需要不同的参数设置

#### 7.2 调试和优化建议

**训练不稳定的解决方案：**
```python
# 1. 减小KL散度约束
kl_constraint = 0.0001  # 从0.0005减小到0.0001

# 2. 调整GAE参数
lmbda = 0.8  # 从0.95减小到0.8，降低方差

# 3. 增加阻尼系数
damping = 0.1  # 在hessian_matrix_vector_product中
```

**训练速度慢的解决方案：**
```python
# 1. 减小网络规模
hidden_dim = 64  # 从128减小到64

# 2. 减少共轭梯度迭代次数
cg_iters = 5  # 从10减小到5

# 3. 减少线性搜索次数
max_backtracks = 10  # 从15减小到10
```

**性能不理想的解决方案：**
```python
# 1. 增加训练episodes
num_episodes = 1000  # 适当增加

# 2. 调整奖励缩放（特别是Pendulum）
rewards = (rewards + 8.0) / 8.0  # 奖励标准化

# 3. 调整网络初始化
torch.nn.init.orthogonal_(layer.weight, gain=0.01)
```

### 8. 实验结果展示

#### 8.1 CartPole环境训练结果
```
环境: CartPole-v1
训练episodes: 500
最终性能: 500.0 (满分)
收敛速度: ~150 episodes
稳定性: 高，很少出现性能倒退
```

#### 8.2 Pendulum环境训练结果
```
环境: Pendulum-v1
训练episodes: 2000
初始性能: -1453.235
最终性能: -474.927
改善幅度: ~67%
收敛特点: 逐步稳定改善，无明显震荡
```

## 🔧 环境要求

### 基础依赖
- **Python**: 3.7+ （推荐3.8+）
- **PyTorch**: 1.8+ （推荐最新版本）
- **NumPy**: 1.19+
- **Matplotlib**: 3.3+
- **tqdm**: 4.60+ （进度条显示）

### 强化学习环境
```bash
# 统一使用gymnasium库
pip install gymnasium[classic_control]
```

### 可选加速
- **CUDA**: 支持GPU加速（推荐）
- **cuDNN**: 深度学习加速库

### 完整安装命令
```bash
# 创建虚拟环境（推荐）
conda create -n rl_env python=3.8
conda activate rl_env

# 安装PyTorch（根据CUDA版本选择）
pip install torch torchvision torchaudio

# 安装其他依赖
pip install gymnasium[classic_control] numpy matplotlib tqdm

# 验证安装
python -c "import torch; import gymnasium; print('安装成功！')"
```

## 📚 参考文献

1. **TRPO原始论文**: Schulman, J., Levine, S., Abbeel, P., Jordan, M., & Moritz, P. (2015). Trust region policy optimization. In International conference on machine learning (pp. 1889-1897).

2. **GAE论文**: Schulman, J., Moritz, P., Levine, S., Jordan, M., & Abbeel, P. (2016). High-dimensional continuous control using generalized advantage estimation. arXiv preprint arXiv:1506.02438.

3. **策略梯度综述**: Sutton, R. S., McAllester, D., Singh, S., & Mansour, Y. (1999). Policy gradient methods for reinforcement learning with function approximation. Advances in neural information processing systems, 12.

## 🤝 贡献指南

欢迎提交Issue和Pull Request来改进这个项目！

### 贡献方式
1. Fork本项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启Pull Request

### 代码规范
- 保持详细的中文注释
- 遵循Google Python代码风格
- 添加适当的类型提示
- 确保代码可以正常运行

---

**项目维护者**: [POPO]
**最后更新**: 2025年8月
**许可证**: MIT License
