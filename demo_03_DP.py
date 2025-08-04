"""
动态规划 - 策略迭代算法实现悬崖漫步问题
===========================================

本文件实现了强化学习中的经典问题：悬崖漫步（Cliff Walking）
使用动态规划中的策略迭代（Policy Iteration）算法来求解最优策略

算法原理：
- 策略迭代包含两个主要步骤：策略评估和策略提升
- 策略评估：给定策略π，计算其状态价值函数V^π(s)
- 策略提升：根据当前价值函数，贪心地改进策略
- 重复上述过程直到策略收敛到最优策略π*

数学基础：
- 贝尔曼方程：V^π(s) = Σ_a π(a|s) Σ_{s',r} p(s',r|s,a)[r + γV^π(s')]
- 策略提升：π'(s) = argmax_a Σ_{s',r} p(s',r|s,a)[r + γV^π(s')]

作者：[您的姓名]
日期：[当前日期]
"""

import copy  # 用于深拷贝策略，避免引用问题


class CliffWalkingEnv:
    """
    悬崖漫步环境类 - 强化学习经典问题环境实现
    =============================================

    环境描述：
    - 这是一个4×12的网格世界（默认大小）
    - 智能体从左下角(3,0)出发，目标是到达右下角(3,11)
    - 悬崖位于最下面一行除了起点和终点的位置：(3,1)到(3,10)
    - 智能体可以执行4种动作：上(0)、下(1)、左(2)、右(3)

    奖励机制：
    - 每走一步得到-1的奖励（鼓励找到最短路径）
    - 如果智能体掉入悬崖，会得到-100的奖励（严重惩罚）
    - 到达终点后任务结束

    状态表示：
    - 使用一维索引表示二维网格位置：state = row * ncol + col
    - 例如：位置(2,3)对应状态索引 2*12+3=27

    网格布局示例（4×12）：
    [ 0][ 1][ 2]...[ 11]  第0行
    [12][13][14]...[23]   第1行
    [24][25][26]...[35]   第2行
    [36][37][38]...[47]   第3行（起点36，悬崖37-46，终点47）
    """

    def __init__(self, ncol=12, nrow=4):
        """
        初始化悬崖漫步环境

        参数：
            ncol (int): 网格世界的列数（宽度），默认12
            nrow (int): 网格世界的行数（高度），默认4
        """
        self.ncol = ncol  # 定义网格世界的列数（宽度）
        self.nrow = nrow  # 定义网格世界的行数（高度）

        # 状态转移矩阵P的结构说明：
        # P[state][action] = [(probability, next_state, reward, done)]
        # - state: 当前状态索引 (0 到 nrow*ncol-1)
        # - action: 动作索引 (0:上, 1:下, 2:左, 3:右)
        # - probability: 转移概率（在确定性环境中始终为1.0）
        # - next_state: 执行动作后到达的下一个状态
        # - reward: 执行该动作获得的即时奖励
        # - done: 布尔值，表示是否到达终止状态
        self.P = self.createP()  # 创建完整的状态转移矩阵

    def createP(self):
        """
        创建状态转移矩阵P
        ================

        该方法构建完整的马尔可夫决策过程(MDP)的状态转移矩阵。
        对于每个状态-动作对(s,a)，计算所有可能的转移结果。

        返回：
            P (list): 三维列表结构 P[state][action] = [(prob, next_state, reward, done)]
                     - 第一维：状态索引 (0 到 nrow*ncol-1)
                     - 第二维：动作索引 (0:上, 1:下, 2:左, 3:右)
                     - 第三维：转移结果元组列表

        算法逻辑：
        1. 遍历所有状态和动作组合
        2. 对于每个(状态,动作)对，计算：
           - 下一个状态位置（考虑边界约束）
           - 即时奖励（普通移动-1，掉悬崖-100）
           - 是否终止（到达悬崖或终点）
        """
        # 初始化状态转移矩阵：nrow*ncol个状态，每个状态有4个动作
        # P[s][a] 存储在状态s执行动作a的所有可能转移结果
        P = [[[] for j in range(4)] for i in range(self.nrow * self.ncol)]

        # 定义4种动作对应的坐标变化向量
        # 坐标系：原点(0,0)在左上角，x轴向右，y轴向下
        # change[action] = [x方向变化, y方向变化]
        change = [
            [0, -1],  # 动作0：上移，y坐标减1
            [0, 1],   # 动作1：下移，y坐标加1
            [-1, 0],  # 动作2：左移，x坐标减1
            [1, 0]    # 动作3：右移，x坐标加1
        ]

        # 双重循环遍历网格中的每个位置（状态）
        for i in range(self.nrow):  # i是行索引（y坐标），范围[0, nrow-1]
            for j in range(self.ncol):  # j是列索引（x坐标），范围[0, ncol-1]

                # 对当前位置(i,j)的每个可能动作进行状态转移计算
                for a in range(4):  # a是动作索引：0-上，1-下，2-左，3-右

                    # 特殊情况处理：悬崖和终点状态
                    # 条件：位于最下面一行(i == nrow-1) 且 不是起点(j > 0)
                    # 这包括：悬崖位置(3,1)到(3,10) 和 终点位置(3,11)
                    if i == self.nrow - 1 and j > 0:
                        # 在悬崖或终点执行任何动作都会：
                        # - 状态保持不变（吸收状态）
                        # - 奖励为0（已经结束，无额外奖励或惩罚）
                        # - episode终止（done=True）
                        current_state = i * self.ncol + j  # 计算当前状态的一维索引
                        P[current_state][a] = [(1.0, current_state, 0, True)]
                        continue  # 跳过后续的正常状态转移计算

                    # 正常状态的转移计算（非悬崖非终点）
                    # ==========================================

                    # 步骤1：计算执行动作a后的目标位置坐标
                    # 使用min/max函数确保坐标不会超出网格边界（墙壁效应）
                    # 如果智能体试图走出边界，会停留在边界位置
                    next_x = min(self.ncol - 1, max(0, j + change[a][0]))
                    next_y = min(self.nrow - 1, max(0, i + change[a][1]))

                    # 步骤2：将二维坐标(next_y, next_x)转换为一维状态索引
                    # 转换公式：state_index = row * ncol + col
                    next_state = next_y * self.ncol + next_x

                    # 步骤3：设置默认的转移结果
                    reward = -1    # 默认奖励：每移动一步扣1分（鼓励最短路径）
                    done = False   # 默认不终止：继续游戏

                    # 步骤4：检查下一个位置是否为特殊位置（悬崖或终点）
                    # 条件：下一个位置在最下面一行(next_y == nrow-1) 且 不是起点(next_x > 0)
                    if next_y == self.nrow - 1 and next_x > 0:
                        done = True  # 到达悬崖或终点，episode必须结束

                        # 进一步判断是悬崖还是终点
                        if next_x != self.ncol - 1:  # 不是最右侧位置，说明是悬崖
                            reward = -100  # 掉入悬崖的严重惩罚
                        # 如果是最右侧位置(next_x == ncol-1)，则是终点，保持reward=-1

                    # 步骤5：存储当前状态-动作对的完整转移信息
                    # 格式：[(概率, 下一状态, 奖励, 是否终止)]
                    # 概率为1.0表示这是确定性环境（每个动作的结果是确定的）
                    current_state = i * self.ncol + j
                    P[current_state][a] = [(1.0, next_state, reward, done)]

        return P  # 返回完整构建的状态转移矩阵


class PolicyIteration:
    """
    策略迭代算法类 - 动态规划求解MDP最优策略
    =======================================

    策略迭代是动态规划中求解马尔可夫决策过程(MDP)最优策略的经典算法。
    该算法通过交替执行策略评估和策略提升两个步骤，直到收敛到最优策略。

    算法流程：
    1. 初始化：随机策略π₀ 和 价值函数V₀
    2. 策略评估：给定策略πₖ，求解其价值函数V^πₖ
       V^πₖ(s) = Σₐ πₖ(a|s) Σₛ',ᵣ p(s',r|s,a)[r + γV^πₖ(s')]
    3. 策略提升：基于V^πₖ贪心地改进策略
       πₖ₊₁(s) = argmax_a Σₛ',ᵣ p(s',r|s,a)[r + γV^πₖ(s')]
    4. 重复步骤2-3直到策略不再改变

    理论保证：
    - 策略迭代算法保证收敛到最优策略π*
    - 每次迭代策略都不会变差：V^πₖ₊₁ ≥ V^πₖ
    - 在有限状态空间中，算法在有限步内收敛
    """

    def __init__(self, env, theta, gamma):
        """
        初始化策略迭代算法

        参数：
            env: 环境对象（CliffWalkingEnv实例）
            theta (float): 策略评估的收敛阈值，当价值函数变化小于theta时停止
            gamma (float): 折扣因子，范围[0,1]，控制未来奖励的重要性
        """
        self.env = env  # 保存环境引用

        # 初始化状态价值函数V(s)为全零
        # V[s]表示在状态s的价值（期望累积奖励）
        self.v = [0.0] * (self.env.ncol * self.env.nrow)

        # 初始化策略π(a|s)为均匀随机策略
        # pi[s][a]表示在状态s选择动作a的概率
        # 初始时每个动作的概率都是0.25（等概率选择）
        self.pi = [[0.25, 0.25, 0.25, 0.25]
                   for i in range(self.env.ncol * self.env.nrow)]

        self.theta = theta  # 策略评估收敛阈值（通常设为很小的正数，如0.001）
        self.gamma = gamma  # 折扣因子（0表示只关心即时奖励，1表示同等重视未来奖励）

    def policy_evaluation(self):
        """
        策略评估 - 计算给定策略的状态价值函数
        =====================================

        给定当前策略π，通过迭代求解贝尔曼方程来计算状态价值函数V^π(s)。

        贝尔曼方程：
        V^π(s) = Σₐ π(a|s) * Σₛ',ᵣ p(s',r|s,a) * [r + γ * V^π(s') * (1-done)]

        算法步骤：
        1. 初始化新的价值函数new_v
        2. 对每个状态s：
           a) 计算所有动作的Q值：Q^π(s,a) = Σₛ',ᵣ p(s',r|s,a)[r + γV^π(s')]
           b) 根据策略加权求和：V^π(s) = Σₐ π(a|s) * Q^π(s,a)
        3. 检查收敛性：如果max|V_new(s) - V_old(s)| < θ，则停止
        4. 否则更新价值函数，继续迭代

        时间复杂度：O(|S|²|A|) 每次迭代
        """
        cnt = 1  # 迭代轮数计数器

        while True:  # 迭代直到收敛
            max_diff = 0.0  # 记录本轮迭代中价值函数的最大变化量

            # 创建新的价值函数数组，避免在计算过程中修改当前价值函数
            new_v = [0.0] * (self.env.ncol * self.env.nrow)

            # 遍历所有状态，更新每个状态的价值
            for s in range(self.env.ncol * self.env.nrow):
                qsa_list = []  # 存储当前状态s下所有动作的加权Q值

                # 计算状态s下每个动作a的Q值，并按策略概率加权
                for a in range(4):  # 4个动作：上下左右
                    qsa = 0.0  # 动作a的Q值：Q^π(s,a)

                    # 遍历执行动作a可能的所有转移结果
                    # 在确定性环境中，每个(s,a)只有一个转移结果
                    for res in self.env.P[s][a]:
                        p, next_state, r, done = res  # 解包转移结果

                        # 计算Q值：Q^π(s,a) = Σₛ',ᵣ p(s',r|s,a)[r + γV^π(s')]
                        # (1-done)确保终止状态的未来价值为0
                        qsa += p * (r + self.gamma * self.v[next_state] * (1 - done))

                    # 将Q值按策略概率π(a|s)加权，添加到列表中
                    qsa_list.append(self.pi[s][a] * qsa)

                # 状态价值函数：V^π(s) = Σₐ π(a|s) * Q^π(s,a)
                new_v[s] = sum(qsa_list)

                # 更新最大变化量，用于判断收敛
                max_diff = max(max_diff, abs(new_v[s] - self.v[s]))

            # 更新价值函数
            self.v = new_v

            # 检查收敛条件：如果最大变化量小于阈值，则认为已收敛
            if max_diff < self.theta:
                break

            cnt += 1  # 增加迭代计数

        print(f"策略评估进行{cnt}轮后完成，最终收敛精度：{max_diff:.6f}")

    def policy_improvement(self):
        """
        策略提升 - 基于当前价值函数贪心地改进策略
        =========================================

        根据当前的状态价值函数V^π，通过贪心选择来改进策略。
        新策略π'在每个状态选择具有最大动作价值的动作。

        策略提升定理：
        如果π'(s) = argmax_a Σₛ',ᵣ p(s',r|s,a)[r + γV^π(s')]，
        那么V^π'(s) ≥ V^π(s) 对所有状态s成立。

        算法步骤：
        1. 对每个状态s：
           a) 计算所有动作的Q值：Q^π(s,a) = Σₛ',ᵣ p(s',r|s,a)[r + γV^π(s')]
           b) 找到最大Q值对应的动作集合
           c) 在最优动作间均匀分配概率（处理平局情况）
        2. 更新策略π'(a|s)

        注意：如果多个动作具有相同的最大Q值，则在这些动作间均匀分配概率。
        这种处理方式保证了策略的确定性收敛。
        """

        # 遍历所有状态，为每个状态计算改进后的策略
        for s in range(self.env.nrow * self.env.ncol):
            qsa_list = []  # 存储状态s下所有动作的Q值

            # 计算状态s下每个动作的Q值（动作价值函数）
            for a in range(4):  # 4个动作：上下左右
                qsa = 0.0  # 初始化动作a的Q值

                # 计算Q^π(s,a) = Σₛ',ᵣ p(s',r|s,a)[r + γV^π(s')]
                for res in self.env.P[s][a]:
                    p, next_state, r, done = res  # 解包转移结果

                    # 累加期望奖励：即时奖励 + 折扣未来价值
                    qsa += p * (r + self.gamma * self.v[next_state] * (1 - done))

                qsa_list.append(qsa)  # 保存动作a的Q值

            # 策略提升：选择Q值最大的动作
            maxq = max(qsa_list)  # 找到最大Q值
            cntq = qsa_list.count(maxq)  # 统计有多少个动作达到最大Q值

            # 更新策略：在最优动作间均匀分配概率
            # 如果只有一个最优动作，该动作概率为1，其他为0
            # 如果有多个最优动作，在它们之间均匀分配概率
            self.pi[s] = [1.0 / cntq if q == maxq else 0.0 for q in qsa_list]

        print("策略提升完成")
        return self.pi  # 返回改进后的策略

    def policy_iteration(self):
        """
        策略迭代主循环 - 交替执行策略评估和策略提升直到收敛
        =====================================================

        这是策略迭代算法的核心方法，通过反复执行以下步骤直到策略收敛：
        1. 策略评估：计算当前策略π的价值函数V^π
        2. 策略提升：基于V^π贪心地改进策略得到π'
        3. 检查收敛：如果π' = π，则算法收敛，输出最优策略

        收敛性保证：
        - 在有限状态和动作空间中，策略迭代保证在有限步内收敛
        - 收敛到的策略是最优策略π*
        - 每次迭代策略都不会变差（单调改进性质）

        算法复杂度：
        - 时间复杂度：O(k * |S|² * |A|)，其中k是迭代次数
        - 空间复杂度：O(|S| * |A|)
        """
        iteration_count = 0  # 策略迭代轮数计数器

        while True:
            iteration_count += 1
            print(f"\n=== 策略迭代第{iteration_count}轮 ===")

            # 步骤1：策略评估 - 计算当前策略的价值函数
            self.policy_evaluation()

            # 步骤2：保存当前策略的深拷贝，用于后续比较
            # 使用深拷贝避免引用问题，确保比较的准确性
            old_pi = copy.deepcopy(self.pi)

            # 步骤3：策略提升 - 基于当前价值函数改进策略
            new_pi = self.policy_improvement()

            # 步骤4：检查策略是否收敛
            # 如果新策略与旧策略完全相同，说明已找到最优策略
            if old_pi == new_pi:
                print(f"\n策略迭代在第{iteration_count}轮后收敛！")
                print("已找到最优策略π*")
                break
            else:
                print(f"策略在第{iteration_count}轮后发生了改变，继续迭代...")


def print_agent(agent, action_meaning, disaster=[], end=[]):
    """
    可视化智能体的学习结果 - 打印状态价值函数和最优策略
    =====================================================

    该函数以网格形式展示智能体学习到的状态价值和策略，便于分析和理解。

    参数：
        agent: PolicyIteration实例，包含学习到的价值函数和策略
        action_meaning (list): 动作符号列表，如['^', 'v', '<', '>']表示上下左右
        disaster (list): 灾难状态列表（悬崖位置），用于特殊标记
        end (list): 终止状态列表（目标位置），用于特殊标记

    输出格式：
        状态价值：显示每个状态的V(s)值，保留3位小数
        策略：显示每个状态的最优动作，用符号表示
               - 正常状态：显示可选动作（概率>0的动作用符号表示，其他用'o'）
               - 悬崖状态：显示'****'
               - 目标状态：显示'EEEE'
    """

    # 第一部分：打印状态价值函数V(s)
    print("\n" + "="*50)
    print("状态价值函数 V^π(s)：")
    print("="*50)

    for i in range(agent.env.nrow):  # 遍历每一行
        for j in range(agent.env.ncol):  # 遍历每一列
            state_index = i * agent.env.ncol + j  # 计算状态索引
            value = agent.v[state_index]  # 获取状态价值

            # 格式化输出：保持6个字符宽度，显示3位小数
            print('%6.3f' % value, end=' ')
        print()  # 换行

    # 第二部分：打印策略π(a|s)
    print("\n" + "="*50)
    print("最优策略 π*(a|s)：")
    print("="*50)
    print("符号说明：'^'=上, 'v'=下, '<'=左, '>'=右, 'o'=不选择")
    print("特殊标记：'****'=悬崖, 'EEEE'=终点")
    print("-"*50)

    for i in range(agent.env.nrow):  # 遍历每一行
        for j in range(agent.env.ncol):  # 遍历每一列
            state_index = i * agent.env.ncol + j  # 计算状态索引

            # 处理特殊状态的显示
            if state_index in disaster:  # 悬崖状态
                print('****', end=' ')
            elif state_index in end:  # 目标状态
                print('EEEE', end=' ')
            else:  # 普通状态
                # 获取当前状态的策略概率分布
                policy_probs = agent.pi[state_index]
                pi_str = ''

                # 构建策略字符串：概率>0的动作显示对应符号，否则显示'o'
                for k in range(len(action_meaning)):
                    if policy_probs[k] > 0:  # 该动作被选择（概率>0）
                        pi_str += action_meaning[k]
                    else:  # 该动作不被选择
                        pi_str += 'o'

                print(pi_str, end=' ')
        print()  # 换行

    print("="*50)



# ============================================================================
# 主程序执行部分 - 悬崖漫步问题的策略迭代求解
# ============================================================================

if __name__ == "__main__":
    """
    主程序：使用策略迭代算法求解悬崖漫步问题

    实验设置：
    - 环境：4×12网格的悬崖漫步环境
    - 算法：策略迭代（Policy Iteration）
    - 参数：收敛阈值θ=0.001，折扣因子γ=0.9

    预期结果：
    - 智能体学会避开悬崖，选择安全路径到达目标
    - 最优策略通常是向上绕行，避免掉入悬崖
    """

    print("悬崖漫步问题 - 策略迭代算法求解")
    print("="*60)

    # 步骤1：创建悬崖漫步环境
    env = CliffWalkingEnv()
    print(f"环境设置：{env.nrow}×{env.ncol}网格世界")
    print(f"起点：位置({env.nrow-1}, 0)，状态索引{(env.nrow-1)*env.ncol + 0}")
    print(f"终点：位置({env.nrow-1}, {env.ncol-1})，状态索引{(env.nrow-1)*env.ncol + (env.ncol-1)}")
    print(f"悬崖：位置({env.nrow-1}, 1)到({env.nrow-1}, {env.ncol-2})")

    # 步骤2：定义动作符号映射
    action_meaning = ['^', 'v', '<', '>']  # 上、下、左、右动作的可视化符号
    print(f"动作定义：{dict(enumerate(action_meaning))}")

    # 步骤3：设置算法参数
    theta = 0.001   # 策略评估收敛阈值：当价值函数变化小于此值时认为收敛
    gamma = 0.9     # 折扣因子：控制未来奖励的重要性，0.9表示较重视未来奖励
    print(f"算法参数：收敛阈值θ={theta}, 折扣因子γ={gamma}")

    # 步骤4：创建策略迭代智能体并开始学习
    print("\n开始策略迭代学习...")
    agent = PolicyIteration(env, theta, gamma)
    agent.policy_iteration()  # 执行策略迭代算法

    # 步骤5：可视化学习结果
    print("\n学习完成！以下是最终结果：")

    # 定义特殊状态用于可视化
    disaster_states = list(range(37, 47))  # 悬崖状态：索引37-46
    end_states = [47]  # 目标状态：索引47

    # 打印状态价值函数和最优策略
    print_agent(agent, action_meaning, disaster_states, end_states)

    # 步骤6：分析结果
    print("\n结果分析：")
    print("- 状态价值：数值越大表示该位置越有利")
    print("- 最优策略：智能体应该选择的动作方向")
    print("- 预期行为：智能体会选择安全路径（通常向上绕行）避开悬崖")
    print("- 悬崖附近的状态价值较低，体现了掉入悬崖的风险")



# ============================================================================
# 价值迭代算法实现 - 动态规划的另一种求解方法
# ============================================================================

class ValueIteration:
    """
    价值迭代算法类 - 直接求解最优价值函数的动态规划方法
    =====================================================

    价值迭代是求解马尔可夫决策过程(MDP)最优策略的另一种动态规划算法。
    与策略迭代不同，价值迭代直接迭代价值函数，无需显式的策略评估步骤。

    算法原理：
    价值迭代基于贝尔曼最优方程进行迭代更新：
    V_{k+1}(s) = max_a Σ_{s',r} p(s',r|s,a)[r + γV_k(s')]

    与策略迭代的区别：
    1. 策略迭代：交替进行策略评估和策略提升，每次策略评估需要多轮迭代
    2. 价值迭代：直接更新价值函数，每轮只需一次更新，最后提取策略

    收敛性：
    - 在γ < 1的条件下，价值迭代保证收敛到最优价值函数V*
    - 收敛速度：O(γ^k)，其中k是迭代次数
    - 通常比策略迭代需要更多轮次，但每轮计算更简单

    适用场景：
    - 当策略评估代价较高时，价值迭代更高效
    - 适合在线学习和实时决策场景
    """

    def __init__(self, env, theta, gamma):
        """
        初始化价值迭代算法

        参数：
            env: 环境对象（CliffWalkingEnv实例）
            theta (float): 价值函数收敛阈值，当价值变化小于theta时停止迭代
            gamma (float): 折扣因子，范围[0,1]，控制未来奖励的重要性
        """
        self.env = env  # 保存环境引用

        # 初始化状态价值函数V(s)为全零
        # 注意：价值迭代直接优化价值函数，不需要维护显式策略
        self.v = [0.0] * (self.env.ncol * self.env.nrow)

        self.theta = theta  # 价值函数收敛阈值
        self.gamma = gamma  # 折扣因子

        # 价值迭代完成后提取的最优策略
        # 初始化为None，在价值迭代结束后通过get_policy()方法计算
        self.pi = [None for i in range(self.env.ncol * self.env.nrow)]

    def value_iteration(self):
        """
        价值迭代主算法 - 直接迭代最优价值函数
        =====================================

        该方法实现价值迭代的核心算法，通过重复应用贝尔曼最优算子来更新价值函数。

        贝尔曼最优方程：
        V*(s) = max_a Σ_{s',r} p(s',r|s,a)[r + γV*(s')]

        算法步骤：
        1. 对每个状态s，计算所有动作的Q值：Q(s,a) = Σ_{s',r} p(s',r|s,a)[r + γV(s')]
        2. 更新价值函数：V_{k+1}(s) = max_a Q(s,a)  ← 关键区别于策略迭代
        3. 检查收敛：如果max_s |V_{k+1}(s) - V_k(s)| < θ，则停止
        4. 否则继续迭代

        与策略迭代的核心区别：
        - 策略迭代：V^π(s) = Σ_a π(a|s) * Q^π(s,a)  (按策略加权平均)
        - 价值迭代：V*(s) = max_a Q*(s,a)           (直接取最大值)

        时间复杂度：O(k * |S| * |A|)，其中k是迭代次数
        """
        cnt = 0  # 迭代轮数计数器

        while True:  # 迭代直到收敛
            max_diff = 0.0  # 记录本轮迭代中价值函数的最大变化量

            # 创建新的价值函数数组，避免在计算过程中修改当前价值函数
            new_v = [0.0] * (self.env.ncol * self.env.nrow)

            # 遍历所有状态，应用贝尔曼最优算子
            for s in range(self.env.ncol * self.env.nrow):
                qsa_list = []  # 存储状态s下所有动作的Q值

                # 计算状态s下每个动作的Q值
                for a in range(4):  # 4个动作：上下左右
                    qsa = 0.0  # 初始化动作a的Q值

                    # 计算Q(s,a) = Σ_{s',r} p(s',r|s,a)[r + γV(s')]
                    for res in self.env.P[s][a]:
                        p, next_state, r, done = res  # 解包转移结果

                        # 累加期望奖励：即时奖励 + 折扣未来价值
                        qsa += p * (r + self.gamma * self.v[next_state] * (1 - done))

                    qsa_list.append(qsa)  # 保存动作a的Q值

                # 【关键区别】价值迭代：直接取所有动作Q值的最大值
                # 这里体现了贝尔曼最优方程：V*(s) = max_a Q*(s,a)
                # 而策略迭代是：V^π(s) = Σ_a π(a|s) * Q^π(s,a)
                new_v[s] = max(qsa_list)

                # 更新最大变化量，用于判断收敛
                max_diff = max(max_diff, abs(new_v[s] - self.v[s]))

            # 更新价值函数
            self.v = new_v

            # 检查收敛条件：如果最大变化量小于阈值，则认为已收敛
            if max_diff < self.theta:
                break

            cnt += 1  # 增加迭代计数

        print(f"价值迭代一共进行{cnt}轮后收敛，最终收敛精度：{max_diff:.6f}")

        # 价值迭代完成后，根据最优价值函数提取最优策略
        self.get_policy()

    def get_policy(self):
        """
        策略提取 - 从最优价值函数中提取最优策略
        =========================================

        价值迭代算法的最后一步：根据收敛的最优价值函数V*，
        通过贪心选择提取最优策略π*。

        策略提取公式：
        π*(s) = argmax_a Σ_{s',r} p(s',r|s,a)[r + γV*(s')]

        算法步骤：
        1. 对每个状态s，重新计算所有动作的Q值（基于最优价值函数V*）
        2. 选择Q值最大的动作作为最优动作
        3. 如果有多个动作具有相同的最大Q值，在它们之间均匀分配概率

        注意事项：
        - 这一步的计算与价值迭代中的Q值计算完全相同
        - 区别在于这里是为了提取策略，而不是更新价值函数
        - 提取的策略是确定性的（或在平局时是均匀随机的）

        时间复杂度：O(|S| * |A|)
        """

        # 遍历所有状态，为每个状态提取最优策略
        for s in range(self.env.nrow * self.env.ncol):
            qsa_list = []  # 存储状态s下所有动作的Q值

            # 基于最优价值函数V*计算每个动作的Q值
            for a in range(4):  # 4个动作：上下左右
                qsa = 0.0  # 初始化动作a的Q值

                # 计算Q*(s,a) = Σ_{s',r} p(s',r|s,a)[r + γV*(s')]
                # 注意：这里使用的是已经收敛的最优价值函数self.v
                for res in self.env.P[s][a]:
                    p, next_state, r, done = res  # 解包转移结果

                    # 累加期望奖励：即时奖励 + 折扣未来最优价值
                    qsa += p * (r + self.gamma * self.v[next_state] * (1 - done))

                qsa_list.append(qsa)  # 保存动作a的Q值

            # 策略提取：选择Q值最大的动作
            # 这里的qsa_list有四个动作，也就是说从4个动作中选取进行操作
            maxq = max(qsa_list)  # 找到最大Q值
            cntq = qsa_list.count(maxq)  # 统计有多少个动作达到最大Q值

            # 构建最优策略：
            # - 如果只有一个最优动作，该动作概率为1，其他为0
            # - 如果有多个最优动作，在它们之间均匀分配概率（处理平局）
            self.pi[s] = [1.0 / cntq if q == maxq else 0.0 for q in qsa_list]

        print("最优策略提取完成")



# ============================================================================
# 价值迭代算法测试部分 - 与策略迭代结果对比
# ============================================================================

print("\n" + "="*80)
print("价值迭代算法测试")
print("="*80)

# 创建相同的环境和参数设置，便于与策略迭代结果对比
env_vi = CliffWalkingEnv()
action_meaning = ['^', 'v', '<', '>']
theta = 0.001  # 使用相同的收敛阈值
gamma = 0.9    # 使用相同的折扣因子

print(f"环境设置：{env_vi.nrow}×{env_vi.ncol}网格世界")
print(f"算法参数：收敛阈值θ={theta}, 折扣因子γ={gamma}")

# 创建价值迭代智能体并开始学习
print("\n开始价值迭代学习...")
agent_vi = ValueIteration(env_vi, theta, gamma)
agent_vi.value_iteration()  # 执行价值迭代算法

# 可视化价值迭代的学习结果
print("\n价值迭代学习完成！以下是最终结果：")

# 定义特殊状态用于可视化（与策略迭代保持一致）
disaster_states = list(range(37, 47))  # 悬崖状态：索引37-46
end_states = [47]  # 目标状态：索引47

# 打印价值迭代的状态价值函数和最优策略
print_agent(agent_vi, action_meaning, disaster_states, end_states)

# 算法比较分析
print("\n" + "="*80)
print("策略迭代 vs 价值迭代 算法比较")
print("="*80)
print("相同点：")
print("- 都基于动态规划思想，保证收敛到最优策略")
print("- 都使用贝尔曼方程进行价值更新")
print("- 最终得到的最优策略和价值函数相同")

print("\n不同点：")
print("- 策略迭代：交替进行策略评估和策略提升，每轮策略评估需要多次迭代")
print("- 价值迭代：直接迭代价值函数，每轮只需一次更新")
print("- 策略迭代：通常需要较少的轮数，但每轮计算量大")
print("- 价值迭代：通常需要较多的轮数，但每轮计算量小")

print("\n适用场景：")
print("- 策略迭代：适合策略评估代价较低的场景")
print("- 价值迭代：适合在线学习和实时决策场景")
print("="*80)


# ============================================================================
# 使用OpenAI Gym的FrozenLake环境测试策略迭代算法
# ============================================================================

print("\n" + "="*80)
print("FrozenLake环境 - 策略迭代算法测试")
print("="*80)

import gymnasium as gym  # 统一使用gymnasium库

# 步骤1：创建FrozenLake环境
env = gym.make("FrozenLake-v1")  # 创建冰湖环境（4x4网格）
env = env.unwrapped  # 解封装以访问内部状态转移矩阵P和其他属性
env.render()  # 渲染环境，显示网格布局（S=起点，F=冰面，H=冰洞，G=目标）

print("\nFrozenLake环境说明：")
print("- S (Start): 起始位置")
print("- F (Frozen): 安全的冰面，可以行走")
print("- H (Hole): 冰洞，掉入后游戏结束")
print("- G (Goal): 目标位置，到达后获得奖励")

# 步骤2：分析环境的状态转移矩阵，识别特殊状态
print("\n正在分析环境的状态转移矩阵...")

holes = set()  # 存储冰洞状态的集合
ends = set()   # 存储目标状态的集合

# 遍历所有状态的转移信息来识别冰洞和目标
for s in env.P:  # s是状态索引
    for a in env.P[s]:  # a是动作索引
        for transition in env.P[s][a]:  # transition是转移结果元组
            # 解析转移结果：(概率, 下一状态, 奖励, 是否终止)
            prob, next_state, reward, done = transition

            if reward == 1.0:  # 奖励为1表示到达目标状态
                ends.add(next_state)
            if done == True:  # 游戏终止的状态（包括冰洞和目标）
                holes.add(next_state)

# 从终止状态中排除目标状态，剩下的就是冰洞状态
holes = holes - ends

print(f"冰洞的状态索引: {sorted(list(holes))}")
print(f"目标的状态索引: {sorted(list(ends))}")

# 步骤3：查看特定状态的转移信息（调试和理解用）
print(f"\n查看状态14（目标左边一格）的状态转移信息：")
for a in env.P[14]:  # 遍历状态14的所有动作
    print(f"动作{a}: {env.P[14][a]}")
    # 输出格式：[(概率, 下一状态, 奖励, 是否终止), ...]

print("\n注意：FrozenLake是随机环境，执行动作时有概率滑向其他方向！")

# 步骤4：使用策略迭代算法求解FrozenLake环境
print("\n" + "-"*60)
print("开始在FrozenLake环境中运行策略迭代算法...")
print("-"*60)

# 定义动作符号映射（Gym库FrozenLake环境的标准定义）
action_meaning = ['<', 'v', '>', '^']  # 左、下、右、上
print(f"动作定义：{dict(enumerate(action_meaning))}")

# 设置算法参数
theta = 1e-5  # 更严格的收敛阈值（比悬崖漫步更小）
gamma = 0.9   # 折扣因子
print(f"算法参数：收敛阈值θ={theta}, 折扣因子γ={gamma}")

# 创建策略迭代智能体并开始学习
agent = PolicyIteration(env, theta, gamma)
agent.policy_iteration()  # 执行策略迭代算法

# 步骤5：可视化学习结果
print("\nFrozenLake策略迭代学习完成！以下是最终结果：")

# 使用已识别的冰洞和目标状态进行可视化
# 注意：这里直接使用状态索引，因为FrozenLake是4x4=16个状态
print_agent(agent, action_meaning, [5, 7, 11, 12], [15])

print("\n结果分析：")
print("- 智能体学会了避开冰洞，寻找到达目标的安全路径")
print("- 由于环境的随机性，策略可能包含多个方向以应对滑动")
print("- 状态价值反映了从每个位置到达目标的期望累积奖励")
print("="*80)


