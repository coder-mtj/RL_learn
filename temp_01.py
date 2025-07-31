import random
import gymnasium as gym
import numpy as np
import collections
from tqdm import tqdm
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import rl_utils
# 移除导入以避免冲突，使用本地定义的变量


class ReplayBuffer:
    """
    经验回放池存放<state, action, reward, next_state, done>的集合

    我们先来考虑这个这个类的作用，显然是放入真实env中的条目

    因此我们需要一个容器来装这下条目，这里我们选择collections.deque带最大容量的双端队列
    然后就是定义成员函数：存储函数，随机取出函数
    """
    def __init__(self, capacity):
        self.capacity = capacity
        # 带有最大容量的双端队列，最早的自动丢弃
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        # 我们将单步经验放入buffer中
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        # 使用ramdom的sample函数，放入容器和数量，并随机抽取
        transitions = random.sample(self.buffer, batch_size)

        states, actions, rewards, next_states, dones = zip(*transitions)

        # 这里因为只有state需要放入NN进行计算，因此这里只有states需要转为numpy格式
        return np.array(states), actions, rewards, np.array(next_states), dones

    def size(self):
        return len(self.buffer)



class Qnet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Qnet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)


    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x



class DQN:
    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, gamma, epsilon, target_update_num, device):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.target_update_num = target_update_num
        self.device = device
        # 定义主Q网络
        self.q_net = Qnet(state_dim, hidden_dim, action_dim).to(self.device)
        # 定义目标网络
        self.target_net = Qnet(state_dim, hidden_dim, action_dim).to(self.device)
        #
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=self.learning_rate)


        self.count = 0


    def take_action(self, state):
        # ε-greedy 策略选择动作
        if np.random.random() < self.epsilon:
            # exploration
            action = np.random.randint(self.action_dim)
        else:
            # exploitation
            # 这里将state转为tensor然后通过qnet计算后argmax
            state = torch.tensor([state], dtype=torch.float).to(self.device)
            action = self.q_net(state).argmax().item()

        return action

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions'], dtype=torch.long).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).to(self.device)

        # 计算主网络估计值
        q_values = self.q_net(states).gather(1, actions)

        # 计算目标网路估计值
        max_next_q_values = self.target_net(next_states).max(1)[0].view(-1, 1)
        #
        q_target = rewards + self.gamma * max_next_q_values * (1 - dones)
        # 计算loss
        loss = torch.mean(F.mse_loss(q_values, q_target))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 定期同步主网络参数到目标网络
        if self.count % self.target_update_num == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
        self.count += 1


lr = 2e-3
num_episodes = 500
hidden_dim = 256
gamma = 0.99
epsilon = 0.1
target_update_num = 10
buffer_size = 10000
minimal_size = 500
batch_size = 64
device= torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 创建CartPole env
env_name = 'CartPole-v1'
env = gym.make(env_name)



replay_buffer = ReplayBuffer(capacity=buffer_size)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
agent = DQN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon, target_update_num, device)


return_list = []
for i in range(10):
    with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
        for i_episode in range(int(num_episodes / 10)):
            episode_reward = 0.0
            state, _ = env.reset()

            done = False
            while not done:
                action = agent.take_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                replay_buffer.add(state, action, reward, next_state, terminated)
                state = next_state
                episode_reward += reward


                if replay_buffer.size() > minimal_size:
                    b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)

                    transition_dict = {
                        'states': b_s,
                        'actions': b_a,
                        'rewards': b_r,
                        'next_states': b_ns,
                        'dones': b_d
                    }

                    agent.update(transition_dict)

            return_list.append(episode_reward)

            if (i_episode + 1) % 10 == 0:
                pbar.set_postfix({
                    'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                    'return': '%.3f' % np.mean(return_list[-10:])
                })
            pbar.update(1)


# 绘制回报曲线
episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('DQN on {}'.format(env_name))
plt.show()

# 绘制滑动平均回报曲线（平滑曲线）
mv_return = rl_utils.moving_average(return_list, 9)  # 窗口大小为9的滑动平均
plt.plot(episodes_list, mv_return)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('DQN on {} (Moving Average)'.format(env_name))
plt.show()










