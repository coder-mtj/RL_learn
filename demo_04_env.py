
"""
时序差分学习 - Sarsa算法实现悬崖漫步问题
==========================================

本文件实现了强化学习中的时序差分(Temporal Difference, TD)学习方法：Sarsa算法
用于求解悬崖漫步问题，展示了从动态规划到无模型强化学习的转变。

算法特点：
- Sarsa是一种在线(on-policy)时序差分学习算法
- 不需要环境的完整模型（状态转移概率矩阵）
- 通过与环境交互来学习最优策略
- 使用ε-贪婪策略平衡探索与利用

核心思想：
- 使用TD误差更新Q值：Q(s,a) ← Q(s,a) + α[r + γQ(s',a') - Q(s,a)]
- 采用当前策略选择的动作a'来更新Q值（on-policy特性）

与动态规划的区别：
- 动态规划：需要完整的环境模型，离线计算
- Sarsa：无需环境模型，在线学习，通过试错获得经验

作者：[您的姓名]
日期：[当前日期]
"""

import matplotlib.pyplot as plt  # 用于绘制学习曲线
import numpy as np              # 数值计算库
from tqdm import tqdm          # 显示训练进度条的库

from demo_03_DP import print_agent


class CliffWalkingEnv:
    """
    悬崖漫步环境类 - 交互式版本
    ===========================

    这是悬崖漫步环境的交互式实现，与之前动态规划版本的区别：
    - 动态规划版本：提供完整的状态转移矩阵P，支持离线规划
    - 交互式版本：只提供step()和reset()接口，支持在线学习

    环境特点：
    - 智能体通过step()方法与环境交互
    - 环境维护当前状态，返回下一状态、奖励和终止信号
    - 支持episode的重置和重新开始

    坐标系统：
    - 原点(0,0)位于左上角
    - x轴向右递增，y轴向下递增
    - 起点：左下角(0, nrow-1)
    - 终点：右下角(ncol-1, nrow-1)
    - 悬崖：底行中间位置(1到ncol-2, nrow-1)
    """

    def __init__(self, ncol, nrow):
        """
        初始化悬崖漫步环境

        参数：
            ncol (int): 网格列数（宽度）
            nrow (int): 网格行数（高度）
        """
        self.nrow = nrow  # 网格行数
        self.ncol = ncol  # 网格列数

        # 智能体当前位置坐标（初始化在起点）
        self.x = 0              # 当前横坐标（列索引）
        self.y = self.nrow - 1  # 当前纵坐标（行索引，起点在左下角）

    def step(self, action):
        """
        执行动作并返回环境反馈
        =====================

        这是强化学习环境的核心接口，智能体通过此方法与环境交互。

        参数：
            action (int): 智能体选择的动作
                         0: 上移  1: 下移  2: 左移  3: 右移

        返回：
            next_state (int): 执行动作后的下一个状态（一维索引）
            reward (float): 即时奖励
            done (bool): 是否到达终止状态

        动作执行逻辑：
        1. 根据动作更新智能体位置坐标
        2. 处理边界约束（撞墙效应）
        3. 计算奖励和终止条件
        4. 返回环境反馈信息
        """

        # 定义4种动作对应的坐标变化
        # 坐标系：原点(0,0)在左上角，x轴向右，y轴向下
        change = [
            [0, -1],  # 动作0：上移，y坐标减1
            [0, 1],   # 动作1：下移，y坐标加1
            [-1, 0],  # 动作2：左移，x坐标减1
            [1, 0]    # 动作3：右移，x坐标加1
        ]

        # 更新智能体位置，使用min/max确保不会越界（墙壁效应）
        self.x = min(self.ncol - 1, max(0, self.x + change[action][0]))
        self.y = min(self.nrow - 1, max(0, self.y + change[action][1]))

        # 将二维坐标转换为一维状态索引
        next_state = self.y * self.ncol + self.x

        # 设置默认奖励和终止状态
        reward = -1    # 每步移动的基本代价
        done = False   # 默认不终止

        # 检查是否到达悬崖或目标（最下面一行，除了起点）
        if self.y == self.nrow - 1 and self.x > 0:
            done = True  # 到达悬崖或目标，episode结束

            # 进一步判断是悬崖还是目标
            if self.x != self.ncol - 1:  # 不是最右侧，说明是悬崖
                reward = -100  # 掉入悬崖的严重惩罚
            # 如果是最右侧(x == ncol-1)，则是目标，保持reward=-1

        # 注意之类的step返回的是next_state, reword, done分别代表下一个状态的序号，进入此状态的奖励以及是否是终点或悬崖
        return next_state, reward, done

    def reset(self):
        """
        重置环境到初始状态
        ================

        在每个episode开始时调用，将智能体位置重置到起点。
        这是强化学习环境的标准接口之一。

        返回：
            initial_state (int): 初始状态的一维索引

        重置逻辑：
        - 将智能体位置设置为左下角起点(0, nrow-1)
        - 返回对应的状态索引
        """
        # 重置智能体位置到起点（左下角）
        self.x = 0              # 起点横坐标
        self.y = self.nrow - 1  # 起点纵坐标（最下面一行）

        # 返回初始状态的一维索引
        return self.y * self.ncol + self.x

#
# class Sarsa:
#     """
#     Sarsa算法类 - 在线时序差分学习
#     ==============================
#
#     Sarsa (State-Action-Reward-State-Action) 是一种在线时序差分学习算法。
#     算法名称来源于其更新序列：(s, a, r, s', a')
#
#     算法特点：
#     - On-policy：使用当前策略生成的经验来改进同一策略
#     - 时序差分：使用TD误差进行增量式学习
#     - 无模型：不需要环境的状态转移概率，通过交互学习
#
#     核心更新公式：
#     Q(s,a) ← Q(s,a) + α[r + γQ(s',a') - Q(s,a)]
#
#     其中：
#     - α: 学习率，控制新信息的接受程度
#     - γ: 折扣因子，控制未来奖励的重要性
#     - TD误差: δ = r + γQ(s',a') - Q(s,a)
#
#     与Q-learning的区别：
#     - Sarsa使用实际选择的动作a'更新Q值（保守）
#     - Q-learning使用最优动作max_a Q(s',a)更新Q值（激进）
#     """
#
#     def __init__(self, ncol, nrow, epsilon, alpha, gamma, n_action=4):
#         """
#         初始化Sarsa算法
#
#         参数：
#             ncol (int): 环境列数
#             nrow (int): 环境行数
#             epsilon (float): ε-贪婪策略参数，探索概率
#             alpha (float): 学习率，范围(0,1]
#             gamma (float): 折扣因子，范围[0,1]
#             n_action (int): 动作空间大小，默认4（上下左右）
#         """
#         # 初始化Q表：Q(s,a)存储状态-动作对的价值估计
#         # 形状：[状态数, 动作数] = [nrow*ncol, n_action]
#         self.Q_table = np.zeros([nrow * ncol, n_action])
#
#         # 算法超参数
#         self.n_action = n_action    # 动作空间大小
#         self.alpha = alpha          # 学习率：控制Q值更新的步长
#         self.gamma = gamma          # 折扣因子：控制未来奖励的权重
#         self.epsilon = epsilon      # 探索概率：ε-贪婪策略的随机性参数
#
#     def take_action(self, state):
#         """
#         ε-贪婪策略选择动作
#         ==================
#
#         ε-贪婪策略是强化学习中平衡探索与利用的经典方法：
#         - 以概率ε随机选择动作（探索）
#         - 以概率(1-ε)选择当前最优动作（利用）
#
#         参数：
#             state (int): 当前状态
#
#         返回：
#             action (int): 选择的动作
#
#         策略特点：
#         - 保证所有动作都有被选择的概率（探索性）
#         - 倾向于选择当前认为最优的动作（利用性）
#         - ε值控制探索程度：ε越大越倾向探索，ε越小越倾向利用
#         """
#         if np.random.random() < self.epsilon:
#             # 探索：随机选择动作
#             action = np.random.randint(self.n_action)
#         else:
#             # 利用：选择Q值最大的动作（贪婪选择）
#             action = np.argmax(self.Q_table[state])
#         return action
#
#     def best_action(self, state):
#         """
#         获取状态的最优动作（用于策略可视化）
#         ================================
#
#         返回当前状态下Q值最大的所有动作。
#         如果多个动作具有相同的最大Q值，都会被标记为最优动作。
#
#         参数：
#             state (int): 查询的状态
#
#         返回：
#             a (list): 长度为n_action的列表，最优动作位置为1，其他为0
#         """
#         Q_max = np.max(self.Q_table[state])  # 找到最大Q值
#         a = [0 for _ in range(self.n_action)]  # 初始化动作标记列表
#
#         # 标记所有具有最大Q值的动作
#         for i in range(self.n_action):
#             if self.Q_table[state, i] == Q_max:
#                 a[i] = 1  # 标记为最优动作
#         return a
#
#     def update(self, s0, a0, r, s1, a1):
#         """
#         Sarsa算法的核心更新规则
#         ======================
#
#         使用时序差分误差更新Q值，这是Sarsa算法的核心。
#
#         参数：
#             s0 (int): 当前状态
#             a0 (int): 当前动作
#             r (float): 即时奖励
#             s1 (int): 下一状态
#             a1 (int): 下一动作（关键：使用实际选择的动作）
#
#         更新公式：
#             TD误差: δ = r + γQ(s',a') - Q(s,a)
#             Q更新: Q(s,a) ← Q(s,a) + α·δ
#
#         Sarsa特点：
#         - 使用实际执行的动作a1来计算TD目标
#         - 这使得Sarsa是on-policy算法（策略一致性）
#         - 相比Q-learning更保守，但更稳定
#         """
#         # 计算TD误差：实际奖励 + 折扣未来价值 - 当前估计价值
#         td_error = r + self.gamma * self.Q_table[s1, a1] - self.Q_table[s0, a0]
#
#         # 使用TD误差更新Q值：Q(s,a) ← Q(s,a) + α·δ
#         self.Q_table[s0, a0] += self.alpha * td_error
#
#
#
# # ============================================================================
# # Sarsa算法训练主程序 - 悬崖漫步问题求解
# # ============================================================================
#
# print("Sarsa算法训练开始")
# print("="*60)
#
# # 步骤1：环境设置
# ncol = 12  # 网格列数
# nrow = 4   # 网格行数
# env = CliffWalkingEnv(ncol, nrow)
# print(f"环境设置：{nrow}×{ncol}悬崖漫步环境")
#
# # 步骤2：设置随机种子以确保结果可复现
# np.random.seed(0)
# print("随机种子已设置，确保实验可复现")
#
# # 步骤3：算法超参数设置
# epsilon = 0.1  # ε-贪婪策略的探索概率（10%探索，90%利用）
# alpha = 0.1    # 学习率（Q值更新的步长）
# gamma = 0.9    # 折扣因子（未来奖励的重要性）
#
# print(f"算法参数：")
# print(f"  - 探索概率 ε = {epsilon}")
# print(f"  - 学习率 α = {alpha}")
# print(f"  - 折扣因子 γ = {gamma}")
#
# # 步骤4：创建Sarsa智能体
# agent = Sarsa(ncol, nrow, epsilon, alpha, gamma)
# print("Sarsa智能体已创建")
#
# # 步骤5：训练设置
# num_episodes = 500  # 训练的episode总数
# print(f"训练设置：{num_episodes}个episodes")
#
# # ============================================================================
# # Sarsa算法训练循环 - 在线学习过程
# # ============================================================================
#
# print("\n开始训练...")
#
# return_list = []  # 记录每个episode的总回报，用于分析学习曲线
#
# # 将训练分成10个阶段，每个阶段显示一个进度条
# for i in range(10):
#     # 使用tqdm显示当前阶段的训练进度
#     with tqdm(total=int(num_episodes / 10), desc='训练阶段 %d' % (i+1)) as pbar:
#
#         # 每个阶段训练 num_episodes/10 个episodes
#         for i_episode in range(int(num_episodes / 10)):
#
#             # ==================== 单个Episode的Sarsa学习流程 ====================
#
#             episode_return = 0  # 记录当前episode的累积回报
#
#             # 步骤1：重置环境，获取初始状态
#             state = env.reset()
#
#             # 步骤2：根据当前策略选择初始动作
#             action = agent.take_action(state)
#
#             done = False  # episode终止标志
#
#             # 步骤3：Sarsa学习循环 - 核心算法实现
#             while not done:
#                 # 3.1 执行动作，观察环境反馈
#                 next_state, reward, done = env.step(action)
#
#                 # 3.2 根据当前策略选择下一个动作
#                 # 注意：这是Sarsa的关键特点，使用策略选择的动作进行更新
#                 next_action = agent.take_action(next_state)
#
#                 # 3.3 累积episode回报（不使用折扣，用于评估性能）
#                 episode_return += reward
#
#                 # 3.4 Sarsa更新：使用(s,a,r,s',a')五元组更新Q值
#                 # 这是算法的核心：Q(s,a) ← Q(s,a) + α[r + γQ(s',a') - Q(s,a)]
#                 agent.update(state, action, reward, next_state, next_action)
#
#                 # 3.5 状态转移：为下一轮迭代做准备
#                 state = next_state      # 当前状态 ← 下一状态
#                 action = next_action    # 当前动作 ← 下一动作
#
#             # ==================== Episode结束处理 ====================
#
#             # 记录本episode的总回报
#             return_list.append(episode_return)
#
#             # 每10个episodes更新一次进度条显示
#             if (i_episode + 1) % 10 == 0:
#                 # 计算最近10个episodes的平均回报
#                 recent_avg_return = np.mean(return_list[-10:])
#
#                 # 更新进度条显示信息
#                 pbar.set_postfix({
#                     'Episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
#                     '平均回报': '%.3f' % recent_avg_return
#                 })
#
#             # 更新进度条
#             pbar.update(1)
#
# # ============================================================================
# # 学习结果可视化 - 分析Sarsa算法的学习曲线
# # ============================================================================
#
# print("\n训练完成！正在生成学习曲线...")
#
# # 准备绘图数据
# episodes_list = list(range(len(return_list)))  # episode编号列表
#
# # 创建学习曲线图
# plt.figure(figsize=(10, 6))
# plt.plot(episodes_list, return_list, linewidth=1, alpha=0.7)
#
# # 添加移动平均线以显示学习趋势
# window_size = 20  # 移动平均窗口大小
# if len(return_list) >= window_size:
#     moving_avg = []
#     for i in range(len(return_list)):
#         if i < window_size - 1:
#             moving_avg.append(np.mean(return_list[:i+1]))
#         else:
#             moving_avg.append(np.mean(return_list[i-window_size+1:i+1]))
#     plt.plot(episodes_list, moving_avg, 'r-', linewidth=2, label=f'{window_size}期移动平均')
#
# # 图表设置
# plt.xlabel('Episodes（训练轮数）', fontsize=12)
# plt.ylabel('Returns（回报）', fontsize=12)
# plt.title('Sarsa算法在悬崖漫步环境中的学习曲线', fontsize=14)
# plt.grid(True, alpha=0.3)
# plt.legend()
#
# # 添加性能分析文本
# final_avg_return = np.mean(return_list[-50:])  # 最后50个episodes的平均回报
# plt.text(0.02, 0.98, f'最终性能（最后50期平均）: {final_avg_return:.2f}',
#          transform=plt.gca().transAxes, verticalalignment='top',
#          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
#
# plt.tight_layout()
# plt.show()
#
# # ============================================================================
# # 学习结果分析
# # ============================================================================
#
# print("\n" + "="*60)
# print("Sarsa算法学习结果分析")
# print("="*60)
#
# print(f"训练episodes总数: {len(return_list)}")
# print(f"初始性能（前10期平均）: {np.mean(return_list[:10]):.2f}")
# print(f"最终性能（后50期平均）: {np.mean(return_list[-50:]):.2f}")
# print(f"最佳单次表现: {max(return_list):.2f}")
# print(f"性能改进: {np.mean(return_list[-50:]) - np.mean(return_list[:10]):.2f}")
#
# print("\n学习特点分析：")
# print("- Sarsa是on-policy算法，学习过程相对保守")
# print("- 由于ε-贪婪策略的探索性，智能体会偶尔选择次优动作")
# print("- 学习曲线可能存在波动，这是探索-利用权衡的体现")
# print("- 最终策略倾向于选择安全路径，避免掉入悬崖")
# print("="*60)
#
#
# class nstep_Sarsa:
#     """ n步Sarsa算法 """
#     def __init__(self, n, ncol, nrow, epsilon, alpha, gamma, n_action=4):
#         self.Q_table = np.zeros([nrow * ncol, n_action])
#         self.n_action = n_action
#         self.alpha = alpha
#         self.gamma = gamma
#         self.epsilon = epsilon
#         self.n = n  # 采用n步Sarsa算法
#         self.state_list = []  # 保存之前的状态
#         self.action_list = []  # 保存之前的动作
#         self.reward_list = []  # 保存之前的奖励
#
#     def take_action(self, state):
#         if np.random.random() < self.epsilon:
#             action = np.random.randint(self.n_action)
#         else:
#             action = np.argmax(self.Q_table[state])
#         return action
#
#     def best_action(self, state):  # 用于打印策略
#         Q_max = np.max(self.Q_table[state])
#         a = [0 for _ in range(self.n_action)]
#         for i in range(self.n_action):
#             if self.Q_table[state, i] == Q_max:
#                 a[i] = 1
#         return a
#
#     def update(self, s0, a0, r, s1, a1, done):
#         self.state_list.append(s0)
#         self.action_list.append(a0)
#         self.reward_list.append(r)
#         if len(self.state_list) == self.n:  # 若保存的数据可以进行n步更新
#             G = self.Q_table[s1, a1]  # 得到Q(s_{t+n}, a_{t+n})
#             for i in reversed(range(self.n)):
#                 G = self.gamma * G + self.reward_list[i]  # 不断向前计算每一步的回报
#                 # 如果到达终止状态,最后几步虽然长度不够n步,也将其进行更新
#                 if done and i > 0:
#                     s = self.state_list[i]
#                     a = self.action_list[i]
#                     self.Q_table[s, a] += self.alpha * (G - self.Q_table[s, a])
#             s = self.state_list.pop(0)  # 将需要更新的状态动作从列表中删除,下次不必更新
#             a = self.action_list.pop(0)
#             self.reward_list.pop(0)
#             # n步Sarsa的主要更新步骤
#             self.Q_table[s, a] += self.alpha * (G - self.Q_table[s, a])
#         if done:  # 如果到达终止状态,即将开始下一条序列,则将列表全清空
#             self.state_list = []
#             self.action_list = []
#             self.reward_list = []
#
#
# np.random.seed(0)
# n_step = 5  # 5步Sarsa算法
# alpha = 0.1
# epsilon = 0.1
# gamma = 0.9
# agent = nstep_Sarsa(n_step, ncol, nrow, epsilon, alpha, gamma)
# num_episodes = 500  # 智能体在环境中运行的序列的数量
#
# return_list = []  # 记录每一条序列的回报
# for i in range(10):  # 显示10个进度条
#     #tqdm的进度条功能
#     with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
#         for i_episode in range(int(num_episodes / 10)):  # 每个进度条的序列数
#             episode_return = 0
#             state = env.reset()
#             action = agent.take_action(state)
#             done = False
#             while not done:
#                 next_state, reward, done = env.step(action)
#                 next_action = agent.take_action(next_state)
#                 episode_return += reward  # 这里回报的计算不进行折扣因子衰减
#                 agent.update(state, action, reward, next_state, next_action,
#                              done)
#                 state = next_state
#                 action = next_action
#             return_list.append(episode_return)
#             if (i_episode + 1) % 10 == 0:  # 每10条序列打印一下这10条序列的平均回报
#                 pbar.set_postfix({
#                     'episode':
#                     '%d' % (num_episodes / 10 * i + i_episode + 1),
#                     'return':
#                     '%.3f' % np.mean(return_list[-10:])
#                 })
#             pbar.update(1)
#
# episodes_list = list(range(len(return_list)))
# plt.plot(episodes_list, return_list)
# plt.xlabel('Episodes')
# plt.ylabel('Returns')
# plt.title('5-step Sarsa on {}'.format('Cliff Walking'))
# plt.show()

class QLearning:
    """ Q-learning算法 """
    def __init__(self, ncol, nrow, epsilon, alpha, gamma, n_action=4):
        self.Q_table = np.zeros([nrow * ncol, n_action])  # 初始化Q(s,a)表格
        self.n_action = n_action  # 动作个数
        self.alpha = alpha  # 学习率
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # epsilon-贪婪策略中的参数

    def take_action(self, state):  #选取下一步的操作
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.n_action)
        else:
            action = np.argmax(self.Q_table[state])
        return action

    def best_action(self, state):  # 用于打印策略
        Q_max = np.max(self.Q_table[state])
        a = [0 for _ in range(self.n_action)]
        for i in range(self.n_action):
            if self.Q_table[state, i] == Q_max:
                a[i] = 1
        return a

    def update(self, s0, a0, r, s1):
        td_error = r + self.gamma * self.Q_table[s1].max(
        ) - self.Q_table[s0, a0]
        self.Q_table[s0, a0] += self.alpha * td_error


np.random.seed(0)
epsilon = 0.1
alpha = 0.1
gamma = 0.9
agent = QLearning(ncol, nrow, epsilon, alpha, gamma)
num_episodes = 500  # 智能体在环境中运行的序列的数量

return_list = []  # 记录每一条序列的回报
for i in range(10):  # 显示10个进度条
    # tqdm的进度条功能
    with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
        for i_episode in range(int(num_episodes / 10)):  # 每个进度条的序列数
            episode_return = 0
            state = env.reset()
            done = False
            while not done:
                action = agent.take_action(state)
                next_state, reward, done = env.step(action)
                episode_return += reward  # 这里回报的计算不进行折扣因子衰减
                agent.update(state, action, reward, next_state)
                state = next_state
            return_list.append(episode_return)
            if (i_episode + 1) % 10 == 0:  # 每10条序列打印一下这10条序列的平均回报
                pbar.set_postfix({
                    'episode':
                    '%d' % (num_episodes / 10 * i + i_episode + 1),
                    'return':
                    '%.3f' % np.mean(return_list[-10:])
                })
            pbar.update(1)

episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('Q-learning on {}'.format('Cliff Walking'))
plt.show()

action_meaning = ['^', 'v', '<', '>']
print('Q-learning算法最终收敛得到的策略为：')
print_agent(agent, env, action_meaning, list(range(37, 47)), [47])




