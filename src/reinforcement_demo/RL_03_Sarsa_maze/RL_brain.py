import numpy as np
import pandas as pd


class RL(object):
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions
        self.learning_rate = learning_rate  # 学习率
        self.gamma = reward_decay  # 奖励衰减
        self.epsilon = e_greedy  # 贪婪度
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def choose_action(self, observation):
        # observation格式类似为：[5.0, 45.0, 35.0, 75.0]
        self.check_state_exist(observation)  # 检测本 state 是否在 q_table 中存在

        # 选择action
        if np.random.uniform() < self.epsilon:  # 贪婪模式，选择Q value最高的action
            state_action = self.q_table.loc[observation, :]

            # 同一个 state, 可能会有多个相同的 Q action value, 所以我们乱序一下
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)

        else:  # 非贪婪模式， 选择随机的 action
            action = np.random.choice(self.actions)

        return action

    def learn(self, *args):
        pass

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            self.q_table = self.q_table.append(
                pd.Series(
                    [0] * len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )


class QLearningTable(RL):
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        super(QLearningTable, self).__init__(actions, learning_rate, reward_decay, e_greedy)

    def learn(self, state, action, R, state_):
        self.check_state_exist(state_)
        q_predict = self.q_table.loc[state, action]
        if state_ != 'terminal':
            q_target = R + self.gamma * self.q_table.loc[state_, :].max()  # 下个 state 不是 终止符
        else:
            q_target = R  # 下个 state 是终止符
        self.q_table.loc[state, action] += self.learning_rate * (q_target - q_predict)  # 更新对应的 state-action 值


class SarsaTable(RL):
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        super(SarsaTable, self).__init__(actions, learning_rate, reward_decay, e_greedy)

    def learn(self, state, action, R, state_, action_):
        self.choose_action(state_)
        q_predict = self.q_table.loc[state, action]
        if state_ != 'terminal':
            q_target = R + self.gamma * self.q_table.loc[state_, action_]  # 下个 state 不是 终止符
        else:
            q_target = R  # 下个 state 是终止符
        self.q_table.loc[state, action] += self.learning_rate * (q_target - q_predict)  # 更新对应的 state-action 值


class SarsaLambdaTbTable(RL):
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9, trace_decay=0.9):
        super(SarsaLambdaTbTable, self).__init__(actions, learning_rate, reward_decay, e_greedy)

        # 后观测算法，eligibility trace
        self.lambda_ = trace_decay
        self.eligibility_trace = self.q_table.copy()

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            to_be_append = pd.Series(
                [0] * len(self.actions),
                index=self.q_table.columns,
                name=state,
            )
            self.q_table = self.q_table.append(to_be_append)

            self.eligibility_trace = self.eligibility_trace.append(to_be_append)

    def learn(self, state, action, R, state_, action_):
        self.choose_action(state_)
        q_predict = self.q_table.loc[state, action]
        if state_ != 'terminal':
            q_target = R + self.gamma * self.q_table.loc[state_, action_]  # 下个 state 不是 终止符
        else:
            q_target = R  # 下个 state 是终止符
        error = (q_target - q_predict)

        # method1
        # self.eligibility_trace.loc[state, action] += 1

        # method2
        self.eligibility_trace.loc[state, :] *= 0
        self.eligibility_trace.loc[state, action] = 1

        # 更新Q表
        self.q_table += self.learning_rate * error * self.eligibility_trace

        # 随着时间衰减 eligibility_trace 的值，离获取reward越远，他的"不可或缺性"越小
        self.eligibility_trace *= self.gamma * self.lambda_
