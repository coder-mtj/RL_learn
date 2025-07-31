import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm  # 用于显示进度条
import random
import time


# 悬崖漫步环境类
class CliffWalkingEnv:
    def __init__(self, ncol, nrow):
        self.nrow = nrow  # 行数
        self.ncol = ncol  # 列数
        self.x = 0  # 记录当前智能体位置的横坐标（列索引）
        self.y = self.nrow - 1  # 记录当前智能体位置的纵坐标（行索引），起点在左下角

    # 执行动作，返回新状态、奖励和是否终止
    def step(self, action):
        # 动作对应的坐标变化：上、下、左、右
        change = [[0, -1], [0, 1], [-1, 0], [1, 0]]
        # 计算新位置，确保不越界
        self.x = min(self.ncol - 1, max(0, self.x + change[action][0]))
        self.y = min(self.nrow - 1, max(0, self.y + change[action][1]))
        next_state = self.y * self.ncol + self.x  # 将坐标转换为状态编号
        reward = -1  # 默认每步奖励为-1（鼓励快速到达目标）
        done = False
        # 判断是否到达悬崖或目标
        if self.y == self.nrow - 1 and self.x > 0:
            done = True
            if self.x != self.ncol - 1:  # 掉入悬崖
                reward = -100
        return next_state, reward, done

    # 重置环境到初始状态
    def reset(self):
        self.x = 0
        self.y = self.nrow - 1
        return self.y * self.ncol + self.x  # 返回初始状态编号


# Dyna-Q算法实现类
class DynaQ:
    def __init__(self, ncol, nrow, epsilon, alpha, gamma, n_planning, n_action=4):
        self.Q_table = np.zeros([nrow * ncol, n_action])  # 初始化Q表
        self.n_action = n_action  # 动作数量
        self.alpha = alpha  # 学习率
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # ε-greedy策略中的探索率

        self.n_planning = n_planning  # 每次Q-learning后执行Q-planning的次数
        self.model = dict()  # 环境模型，存储(s,a)->(r,s')的映射

    # ε-greedy策略选择动作
    def take_action(self, state):
        if np.random.random() < self.epsilon:  # 探索
            action = np.random.randint(self.n_action)
        else:  # 利用
            action = np.argmax(self.Q_table[state])
        return action

    # Q-learning更新
    def q_learning(self, s0, a0, r, s1):
        td_error = r + self.gamma * self.Q_table[s1].max() - self.Q_table[s0, a0]  # 计算TD误差
        self.Q_table[s0, a0] += self.alpha * td_error  # 更新Q值

    # 综合更新：执行Q-learning和Q-planning
    def update(self, s0, a0, r, s1):
        # 1. 执行Q-learning更新真实经验
        self.q_learning(s0, a0, r, s1)
        # 2. 将经验添加到模型中
        self.model[(s0, a0)] = (r, s1)
        # 3. 执行n_planning次Q-planning（基于模型的更新）
        for _ in range(self.n_planning):
            # 随机选择曾经遇到过的状态动作对
            (s, a), (r, s_) = random.choice(list(self.model.items()))
            self.q_learning(s, a, r, s_)  # 使用模型数据更新Q值


# 运行Dyna-Q算法在悬崖漫步环境中的实验
def DynaQ_CliffWalking(n_planning):
    ncol = 12  # 网格列数
    nrow = 4  # 网格行数
    env = CliffWalkingEnv(ncol, nrow)  # 创建环境
    epsilon = 0.01  # ε-greedy参数
    alpha = 0.1  # 学习率
    gamma = 0.9  # 折扣因子
    agent = DynaQ(ncol, nrow, epsilon, alpha, gamma, n_planning)  # 创建Dyna-Q智能体
    num_episodes = 300  # 总训练序列数

    return_list = []  # 记录每个序列的回报
    for i in range(10):  # 分为10个阶段（每个阶段30个序列）显示进度条
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):  # 每个阶段
                episode_return = 0  # 当前序列的累计回报
                state = env.reset()  # 重置环境
                done = False
                while not done:
                    action = agent.take_action(state)  # 选择动作
                    next_state, reward, done = env.step(action)  # 执行动作
                    episode_return += reward  # 累计回报（不折扣）
                    agent.update(state, action, reward, next_state)  # 更新智能体
                    state = next_state  # 状态转移
                return_list.append(episode_return)
                # 每10个序列显示一次平均回报
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({
                        'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                        'return': '%.3f' % np.mean(return_list[-10:])
                    })
                pbar.update(1)
    return return_list


# 主程序
if __name__ == "__main__":
    np.random.seed(0)  # 设置随机种子（可复现结果）
    random.seed(0)
    n_planning_list = [0, 2, 20]  # 对比不同Q-planning步数的影响

    # 运行实验并绘制结果
    for n_planning in n_planning_list:
        print('Q-planning步数为：%d' % n_planning)
        time.sleep(0.5)  # 暂停0.5秒（避免打印混乱）
        return_list = DynaQ_CliffWalking(n_planning)
        episodes_list = list(range(len(return_list)))
        plt.plot(episodes_list, return_list, label=str(n_planning) + ' planning steps')

    # 绘制图表
    plt.legend()
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('Dyna-Q on {}'.format('Cliff Walking'))
    plt.show()