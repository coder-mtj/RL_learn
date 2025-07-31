# 强化学习算法实现项目

本项目包含了多个经典强化学习算法的PyTorch实现，从简单的多臂赌博机到复杂的深度强化学习算法。每个实现都包含详细的注释和说明，适合学习和研究使用。

## 项目结构

- `demo_01_multi_arm_bandit.py`: 多臂赌博机算法实现
- `demo_02_MDP.py`: 马尔可夫决策过程(MDP)的基本概念实现
- `demo_03_DP.py`: 动态规划方法实现
- `demo_04_env.py`: 强化学习环境实现
- `demo_05_dyna_Q.py`: Dyna-Q算法实现
- `demo_06_DQN.py`: 深度Q网络(DQN)算法实现
- `demo_07_double_DQN.py`: Double DQN算法实现
- `demo_08_dueling_DQN.py`: Dueling DQN算法实现
- `demo_09_REINFORCE.py`: REINFORCE (策略梯度)算法实现
- `rl_utils.py`: 强化学习工具函数库

## 环境要求

- Python 3.6+
- PyTorch 1.0+
- NumPy
- Gym
- Matplotlib
- tqdm

可以使用以下命令安装所需依赖：
```bash
pip install torch numpy gym matplotlib tqdm
```

## 算法说明

### 1. 多臂赌博机 (Multi-Armed Bandit)
实现了探索与利用的平衡策略，包括：
- ε-贪婪策略
- UCB算法
- Thompson采样

### 2. 马尔可夫决策过程 (MDP)
包含MDP的基本要素实现：
- 状态转移
- 奖励函数
- 价值函数

### 3. 动态规划 (Dynamic Programming)
实现了基于动态规划的算法：
- 策略迭代
- 价值迭代

### 4. Dyna-Q
实现了基于模型的强化学习算法Dyna-Q，包括：
- Q学习更新
- 模型学习
- 规划更新

### 5. DQN (Deep Q-Network)
实现了DQN算法的核心组件：
- 经验回放
- 目标网络
- ε-贪婪探索

### 6. Double DQN
在DQN的基础上改进，解决过度估计问题：
- 分离动作选择和评估
- 更稳定的Q值估计

### 7. Dueling DQN
实现了基于价值分解的DQN改进：
- 状态价值流
- 动作优势流
- 优势函数归一化

### 8. REINFORCE
实现了基础的策略梯度算法REINFORCE：
- 策略网络直接输出动作概率
- 蒙特卡洛采样估计回报
- 基于轨迹的梯度更新
- 无基线的策略优化

## 使用方法

每个文件都可以独立运行，例如运行REINFORCE算法：
```bash
python demo_09_REINFORCE.py
```

## 关键特性

1. 代码结构清晰，注释详细
2. 实现了完整的训练和评估流程
3. 包含可视化工具，方便分析算法性能
4. 使用PyTorch实现，支持GPU加速
5. 提供详细的算法实现文档

## 实验结果

各算法在典型环境（如CartPole-v1）上都能达到良好的性能：
- DQN和其变体能在CartPole环境中实现稳定控制
- Double DQN能有效缓解Q值过估计问题
- Dueling DQN在某些任务中能获得更好的学习效率
- REINFORCE能直接学习确定性或随机策略

## 贡献指南

欢迎提交Issue和Pull Request来改进代码和文档。在提交代码时请确保：
1. 代码风格符合PEP 8规范
2. 添加适当的注释和文档
3. 确保代码可以正常运行

## 参考资料

- Sutton & Barto的《强化学习：简介》
- Deep Q-Network相关论文
- Policy Gradient方法相关论文
- OpenAI Gym文档
- PyTorch文档

## 许可证

本项目采用MIT许可证。
