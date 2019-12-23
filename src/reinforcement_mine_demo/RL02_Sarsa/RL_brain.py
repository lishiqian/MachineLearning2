import numpy as np
import pandas as pd


class RL(object):

    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions
        self.lr = learning_rate
        self.gamma = reward_decay  # 奖励衰减
        self.epsilon = e_greedy  # 贪婪度

        self.q_table = pd.DataFrame(columns=actions, dtype=np.float64)

    def choose_action(self, observation):
        self.check_state_exist(observation)

        if np.random.uniform() < self.epsilon:  # 贪婪模式
            state_action = self.q_table.loc[observation, :]
            # state_action[state_action == np.max(state_action)].index 的作用为
            # 加入state_action = [1,3,2,3] 那上面代码返回值为列表中最大值得索引即[1, 3]
            return np.random.choice(state_action[state_action == np.max(state_action)].index)
        else:
            return np.random.choice(self.actions)

    def learn(self, *args):
        pass

    def check_state_exist(self, observation):
        if observation not in self.q_table.index:
            self.q_table = self.q_table.append(
                pd.Series(
                    [0] * 4,
                    index=self.q_table.columns,
                    name=observation
                )
            )


class QLearningTable(RL):

    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        super(QLearningTable, self).__init__(actions, learning_rate, reward_decay, e_greedy)

    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)

        q_predict = self.q_table.loc[s, a]
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.loc[s_, :].max()
        else:
            q_target = r
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)


class SarsaTable(RL):

    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        super(SarsaTable, self).__init__(actions, learning_rate, reward_decay, e_greedy)

    def learn(self, s, a, r, s_, a_):
        self.check_state_exist(s_)

        q_predict = self.q_table.loc[s, a]
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.loc[s_, a_]
        else:
            q_target = r
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)


# SarasLambdaTable为批量更新q_table的值
class SarasLambdaTable(RL):

    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9, trace_decay=0.9):
        super(SarasLambdaTable, self).__init__(actions, learning_rate, reward_decay, e_greedy)

        self.lambda_ = trace_decay
        self.eligibility_trace = self.q_table.copy()

    def learn(self, s, a, r, s_, a_):
        self.check_state_exist(s_)

        q_predict = self.q_table.loc[s, a]
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.loc[s_, a_]
        else:
            q_target = r
        error = q_target - q_predict

        self.eligibility_trace.loc[s, :] *= 0
        self.eligibility_trace.loc[s, a] = 1

        # self.eligibility_trace.loc[s, a] += 1.0

        # 批量更新q_table的值
        self.q_table += self.lr * error * self.eligibility_trace
        # 随着时间eligibility_trace的值将衰减
        self.eligibility_trace *= self.gamma * self.lambda_

    def check_state_exist(self, observation):
        if observation not in self.q_table.index:
            to_be_append = pd.Series(
                [0] * 4,
                index=self.q_table.columns,
                name=observation
            )

            self.q_table = self.q_table.append(to_be_append)
            self.eligibility_trace = self.eligibility_trace.append(to_be_append)
