import numpy as np
import pandas as pd


class QLearningTable(object):
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions
        self.lr = learning_rate  # 学习率
        self.gamma = reward_decay  # 奖励衰减
        self.epsilon = e_greedy  # 贪婪度

        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def choose_action(self, observation):
        self._check_obsesrvation_exist(observation)

        if np.random.rand() < self.epsilon:  # 贪婪模式
            state_action = self.q_table.loc[observation, :]

            # 同一个 state, 可能会有多个相同的 Q action value, 所以我们乱序一下
            return np.random.choice(state_action[state_action == np.max(state_action)].index)
        else:  # 探索模式
            return np.random.choice(self.actions)

    def learn(self, s, a, r, s_):
        self._check_obsesrvation_exist(s_)
        q_predict = self.q_table.loc[s, a]
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.loc[s_, :].max()
        else:
            q_target = r
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)

    def _check_obsesrvation_exist(self, observation):
        if observation not in self.q_table.index:
            self.q_table = self.q_table.append(
                pd.Series(
                    [0] * 4,
                    index=self.q_table.columns,
                    name=observation
                )
            )
