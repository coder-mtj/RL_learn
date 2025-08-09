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
- `demo_13_PPO_CartPole.py`: PPO算法在离散动作空间的实现（CartPole环境）
- `demo_14_PPO_Pendulum.py`: PPO算法在连续动作空间的实现（Pendulum环境）

### 高级Actor-Critic方法
- `demo_16_SAC.py`: SAC（Soft Actor-Critic）算法在连续动作空间的实现（Pendulum环境）

### 模仿学习方法
- `demo_17_behavior_cloning.py`: 行为克隆（Behavior Cloning）算法实现，包含PPO专家训练和BC模仿学习对比

## 算法说明

### PPO算法
PPO（Proximal Policy Optimization）是目前最流行的强化学习算法之一，本项目提供了两个版本的实现：

- **demo_13_PPO_CartPole.py**: 离散动作空间版本（CartPole环境）
- **demo_14_PPO_Pendulum.py**: 连续动作空间版本（Pendulum环境）

**PPO损失函数**：
```python
# 截断代理目标函数（Clipped Surrogate Objective）
ratio = torch.exp(log_probs - old_log_probs)  # 重要性采样比率
surr1 = ratio * advantage                     # 第一个代理目标
surr2 = torch.clamp(ratio, 1-eps, 1+eps) * advantage  # 截断代理目标
actor_loss = torch.mean(-torch.min(surr1, surr2))     # PPO策略损失

# 价值函数损失
critic_loss = torch.mean(F.mse_loss(critic(states), td_target))
```

**核心特点**：
- 使用截断机制防止策略更新过大
- 同时优化策略网络和价值网络
- 基于策略梯度和优势函数

### TRPO算法
TRPO（Trust Region Policy Optimization）是一种先进的策略梯度算法，本项目提供了两个版本的实现：

- **demo_11_TRPO_CartPole.py**: 离散动作空间版本（CartPole环境）
- **demo_12_TRPO_pendulum.py**: 连续动作空间版本（Pendulum环境）

### SAC算法
SAC（Soft Actor-Critic）是一种基于最大熵的Actor-Critic算法，特别适用于连续控制任务：

- **demo_16_SAC.py**: 连续动作空间版本（Pendulum环境）

**SAC损失函数**：
```python
# 策略损失（最大熵目标）
entropy = -log_prob.sum(dim=-1, keepdim=True)  # 策略熵
q1_value = critic_1(states, actions)
q2_value = critic_2(states, actions)
actor_loss = torch.mean(-log_alpha.exp() * entropy - torch.min(q1_value, q2_value))

# Q网络损失（双Q学习）
q1_loss = F.mse_loss(q1_value, td_target.detach())
q2_loss = F.mse_loss(q2_value, td_target.detach())

# 温度参数损失（自动调节）
alpha_loss = torch.mean(-log_alpha * (entropy + target_entropy).detach())
```

**核心特点**：
- 最大熵原理：平衡探索与利用
- 双Q网络：减少价值过估计
- 自动温度调节：动态平衡熵与奖励
- 样本效率高、训练稳定、探索能力强

### 行为克隆算法
行为克隆（Behavior Cloning）是一种模仿学习方法，通过监督学习模仿专家行为：

- **demo_17_behavior_cloning.py**: 完整的BC实现，包含PPO专家训练和BC模仿学习

**BC损失函数**：
```python
# 负对数似然损失（最大似然估计）
log_probs = torch.log(policy(states).gather(1, actions))  # 计算对数概率
bc_loss = torch.mean(-log_probs)  # 负对数似然损失

# 等价于交叉熵损失
# bc_loss = F.cross_entropy(policy_logits, actions)
```

**数学原理**：
```
专家数据似然: L(θ) = ∏ P(a_i | s_i, θ)
对数似然: log L(θ) = ∑ log P(a_i | s_i, θ)
损失函数: Loss = -log L(θ) = -∑ log P(a_i | s_i, θ)
```

**核心特点**：
- 监督学习范式：直接学习状态到动作的映射
- 最大似然估计：最大化专家动作的概率
- 单网络结构：只需要策略网络，无需价值网络
- 专家数据依赖：性能受限于专家数据质量和数量
- **应用场景**: 自动驾驶、机器人控制、游戏AI

## 🔍 损失函数对比分析

### 三种算法的损失函数本质区别

| 算法 | 损失函数类型 | 数学基础 | 优化目标 | 网络结构 |
|------|-------------|----------|----------|----------|
| **PPO** | 截断代理目标 | 策略梯度 + 信任域 | 最大化期望回报（有约束） | Actor + Critic |
| **SAC** | 最大熵目标 | 策略梯度 + 最大熵 | 最大化期望回报 + 熵 | Actor + 双Critic |
| **BC** | 负对数似然 | 最大似然估计 | 最大化专家动作概率 | 仅Policy |

### 详细对比

**1. PPO - 强化学习（策略梯度）**
```python
# 核心：限制策略更新幅度，防止性能崩塌
actor_loss = -torch.min(ratio * advantage, clipped_ratio * advantage)
```
- **特点**：通过截断机制确保策略更新稳定
- **适用**：需要与环境交互学习的任务
- **优势**：训练稳定，理论保证

**2. SAC - 强化学习（最大熵）**
```python
# 核心：平衡奖励最大化和策略熵最大化
actor_loss = -torch.min(q1, q2) - alpha * entropy
```
- **特点**：鼓励探索，自动调节探索-利用平衡
- **适用**：连续控制任务，需要高样本效率
- **优势**：样本效率高，探索能力强

**3. BC - 监督学习（模仿学习）**
```python
# 核心：直接模仿专家行为，最大化专家动作概率
bc_loss = -torch.mean(log_probs)  # 等价于交叉熵
```
- **特点**：无需环境交互，直接从专家数据学习
- **适用**：有高质量专家演示的任务
- **优势**：训练简单，无需奖励函数

### 损失函数的数学含义

**PPO损失函数**：
- 目标：`max E[min(r(θ)A, clip(r(θ))A)]`
- 含义：在信任域内最大化优势加权的策略改进

**SAC损失函数**：
- 目标：`max E[R + α·H(π)]`
- 含义：同时最大化奖励和策略熵（探索性）

**BC损失函数**：
- 目标：`max ∏P(a_expert|s)`
- 含义：最大化在专家状态下选择专家动作的概率

## 使用说明

### 环境准备
```bash
# 安装基础依赖
pip install torch numpy matplotlib tqdm

# 安装gymnasium库
pip install gymnasium[classic_control]
```

#### 常见错误
```python
# ❌ 错误：连续算法 + 离散环境
env = gym.make('CartPole-v1')  # 离散动作空间
agent = PPOContinuous(...)     # 连续动作算法

# ✅ 正确：匹配的组合
env = gym.make('CartPole-v1')  # 离散动作空间
agent = PPO(...)               # 离散动作算法

# ✅ 行为克隆正确使用
# 1. 先训练专家（PPO）
# 2. 采样专家数据
# 3. 训练BC智能体
```



## 📚 参考文献

### 策略梯度方法
1. **PPO原始论文**: Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347.

2. **TRPO原始论文**: Schulman, J., Levine, S., Abbeel, P., Jordan, M., & Moritz, P. (2015). Trust region policy optimization. In International conference on machine learning (pp. 1889-1897).

3. **GAE论文**: Schulman, J., Moritz, P., Levine, S., Jordan, M., & Abbeel, P. (2016). High-dimensional continuous control using generalized advantage estimation. arXiv preprint arXiv:1506.02438.

### Actor-Critic方法
4. **SAC原始论文**: Haarnoja, T., Zhou, A., Abbeel, P., & Levine, S. (2018). Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor. In International conference on machine learning (pp. 1861-1870).

5. **SAC改进版本**: Haarnoja, T., Zhou, A., Hartikainen, K., Tucker, G., Ha, S., Tan, J., ... & Levine, S. (2018). Soft actor-critic algorithms and applications. arXiv preprint arXiv:1812.05905.

### 模仿学习方法
6. **行为克隆综述**: Pomerleau, D. A. (1991). Efficient training of artificial neural networks for autonomous navigation. Neural computation, 3(1), 88-97.

7. **模仿学习理论**: Ross, S., Gordon, G., & Bagnell, D. (2011). A reduction of imitation learning and structured prediction to no-regret online learning. In Proceedings of the fourteenth international conference on artificial intelligence and statistics (pp. 627-635).

### 基础理论
8. **策略梯度综述**: Sutton, R. S., McAllester, D., Singh, S., & Mansour, Y. (1999). Policy gradient methods for reinforcement learning with function approximation. Advances in neural information processing systems, 12.

9. **OpenAI实现参考**: Dhariwal, P., Hesse, C., Klimov, O., Nichol, A., Plappert, M., Radford, A., ... & Wu, J. (2017). OpenAI baselines. GitHub repository.

---

## 🚀 最新更新

### v2.1.0 (2025年8月)
- ✅ 新增SAC（Soft Actor-Critic）算法实现
- ✅ 新增行为克隆（Behavior Cloning）算法实现
- ✅ 完善代码注释，每行代码都有详细说明
- ✅ 添加训练进度条和性能可视化
- ✅ 修复gymnasium兼容性问题

### v2.0.0 (2025年7月)
- ✅ 升级到gymnasium库（替代已弃用的gym）
- ✅ 完善PPO和TRPO算法实现
- ✅ 添加连续和离散动作空间支持
- ✅ 优化代码结构和注释

---

**项目维护者**: [POPO]
**最后更新**: 2025年8月
**许可证**: MIT License

## 🤝 贡献指南

欢迎提交Issue和Pull Request！如果你有任何问题或建议，请随时联系。

### 开发计划
- [ ] 添加更多模仿学习算法（GAIL、ValueDice等）
- [ ] 实现多智能体强化学习算法
- [ ] 添加更多环境支持
- [ ] 优化算法性能和稳定性
