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
- `demo_11_TRPO_CartPole.py`: TRPO算法（信任区域策略优化）

## TRPO算法详细说明

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
```python
class PolicyNet(torch.nn.Module):
    """将状态映射为动作概率分布的策略网络"""
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)
```

#### 2.3 价值网络（Critic）
```python
class ValueNet(torch.nn.Module):
    """评估状态价值的网络"""
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)
```

### 3. 算法核心步骤

1. **收集轨迹数据**
   - 使用当前策略与环境交互
   - 存储状态、动作、奖励等信息

2. **计算优势估计**
   - 使用GAE方法计算优势函数值
   - 考虑多步回报和时序差分误差

3. **策略更新**
   - 计算策略梯度
   - 使用共轭梯度法求解二阶优化问题
   - 在KL散度约束下进行线性搜索

4. **价值函数更新**
   - 使用时序差分目标更新价值网络
   - 最小化价值估计的均方误差

### 4. 参数配置

```python
# 网络参数
hidden_dim = 128    # 隐藏层维度

# 训练参数
num_episodes = 500  # 训练回合数
gamma = 0.98       # 折扣因子
lmbda = 0.95      # GAE参数
critic_lr = 1e-2   # 评论家学习率

# TRPO特有参数
kl_constraint = 0.0005  # KL散度约束
alpha = 0.5         # 线性搜索步长
```

### 5. 使用说明

1. **环境准备**
```python
# 安装依赖
pip install torch numpy gym matplotlib tqdm

# 创建环境
env = gym.make('CartPole-v1')
```

2. **训练模型**
```python
# 创建智能体
agent = TRPO(hidden_dim, env.observation_space, env.action_space, 
            lmbda, kl_constraint, alpha, critic_lr, gamma, device)

# 开始训练
return_list = train_on_policy_agent(env, agent, num_episodes)
```

3. **查看结果**
```python
# 绘制训练曲线
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.show()
```

### 6. 优势和特点

1. **稳定性**：通过KL散度约束确保策略更新的保守性
2. **可靠性**：理论上保证单调策略改进
3. **适用性**：适合连续动作空间和离散动作空间
4. **效率**：使用共轭梯度法高效求解二阶优化问题

### 7. 注意事项

1. KL散度约束的设置很关键，过大会导致不稳定，过小会影响学习效率
2. 需要较大的批量数据来准确估计KL散度
3. 计算复杂度相对较高，建议使用GPU加速
4. 对超参数较敏感，可能需要多次调整

### 8. 常见问题解决

1. 训练不稳定
   - 减小KL散度约束
   - 增加批量数据大小
   - 调整GAE的λ参数

2. 训练速度慢
   - 使用GPU加速
   - 减小网络规模
   - 适当增大学习率

3. 性能不理想
   - 检查奖励设计
   - 调整网络架构
   - 增加训练轮数

## 环境要求

- Python 3.6+
- PyTorch 1.0+
- OpenAI Gym
- NumPy
- Matplotlib
- tqdm
- CUDA（推荐）

## 参考文献

1. Schulman, J., Levine, S., Abbeel, P., Jordan, M., & Moritz, P. (2015). Trust region policy optimization. In International conference on machine learning (pp. 1889-1897).
2. Schulman, J., Moritz, P., Levine, S., Jordan, M., & Abbeel, P. (2016). High-dimensional continuous control using generalized advantage estimation. arXiv preprint arXiv:1506.02438.
