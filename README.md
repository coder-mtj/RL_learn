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
- `demo_13_PPO.py`: PPO算法在离散动作空间的实现（CartPole环境）
- `demo_14_PPO_Pendulum.py`: PPO算法在连续动作空间的实现（Pendulum环境）

## 算法说明

### PPO算法
PPO（Proximal Policy Optimization）是目前最流行的强化学习算法之一，本项目提供了两个版本的实现：

- **demo_13_PPO.py**: 离散动作空间版本（CartPole环境）
- **demo_14_PPO_Pendulum.py**: 连续动作空间版本（Pendulum环境）

### TRPO算法
TRPO（Trust Region Policy Optimization）是一种先进的策略梯度算法，本项目提供了两个版本的实现：

- **demo_11_TRPO_CartPole.py**: 离散动作空间版本（CartPole环境）
- **demo_12_TRPO_pendulum.py**: 连续动作空间版本（Pendulum环境）

## 使用说明

### 环境准备
```bash
# 安装基础依赖
pip install torch numpy matplotlib tqdm

# 安装gymnasium库
pip install gymnasium[classic_control]
```

### 运行算法

**PPO算法**：
```bash
# 离散动作空间版本（CartPole环境）
python demo_13_PPO.py

# 连续动作空间版本（Pendulum环境）
python demo_14_PPO_Pendulum.py
```

**TRPO算法**：
```bash
# 离散动作空间版本（CartPole环境）
python demo_11_TRPO_CartPole.py

# 连续动作空间版本（Pendulum环境）
python demo_12_TRPO_pendulum.py
```

**其他算法**：
```bash
# REINFORCE算法
python demo_09_REINFORCE.py

# Actor-Critic算法
python demo_10_Actor_Critic.py

# DQN系列算法
python demo_06_DQN.py
python demo_07_double_DQN.py
python demo_08_dueling_DQN.py
```

### 性能表现

**CartPole-v1环境（离散动作）**：
- 收敛速度：通常在100-200个episodes内达到满分（500分）
- 训练稳定性：训练过程平稳，很少出现性能大幅波动
- 最终性能：能够稳定达到最大奖励500分

**Pendulum-v1环境（连续动作）**：
- 初始性能：约-1171.6（第50回合）
- 最佳性能：约-625.3（第450回合）
- 改善幅度：约47%的性能提升
- 训练特点：整体呈现改善趋势，但存在一定波动

### 注意事项

#### 环境与算法匹配
- **离散动作空间**：使用 `demo_13_PPO.py` 或 `demo_11_TRPO_CartPole.py`
- **连续动作空间**：使用 `demo_14_PPO_Pendulum.py` 或 `demo_12_TRPO_pendulum.py`
- **动作维度获取**：
  - 离散：`env.action_space.n`
  - 连续：`env.action_space.shape[0]`

#### 常见错误
```python
# ❌ 错误：连续算法 + 离散环境
env = gym.make('CartPole-v1')  # 离散动作空间
agent = PPOContinuous(...)     # 连续动作算法

# ✅ 正确：匹配的组合
env = gym.make('CartPole-v1')  # 离散动作空间
agent = PPO(...)               # 离散动作算法
```



## 📚 参考文献

1. **PPO原始论文**: Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347.

2. **TRPO原始论文**: Schulman, J., Levine, S., Abbeel, P., Jordan, M., & Moritz, P. (2015). Trust region policy optimization. In International conference on machine learning (pp. 1889-1897).

3. **GAE论文**: Schulman, J., Moritz, P., Levine, S., Jordan, M., & Abbeel, P. (2016). High-dimensional continuous control using generalized advantage estimation. arXiv preprint arXiv:1506.02438.

4. **策略梯度综述**: Sutton, R. S., McAllester, D., Singh, S., & Mansour, Y. (1999). Policy gradient methods for reinforcement learning with function approximation. Advances in neural information processing systems, 12.

5. **OpenAI PPO实现**: Dhariwal, P., Hesse, C., Klimov, O., Nichol, A., Plappert, M., Radford, A., ... & Wu, J. (2017). OpenAI baselines. GitHub repository.

---

**项目维护者**: [POPO]
**最后更新**: 2025年8月
**许可证**: MIT License
