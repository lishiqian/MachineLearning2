import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

N_STATES = 6  # 寻找到宝藏的步数
ACTIONS = ['left', 'right']  # 探索者可以选择的动作
EPSILON = 0.9  # 贪婪度 探索者有0.9的概率选择当前步Q表里面的最大值得动作
ALPHA = 0.1  # 学习率
GAMMA = 0.9  # 奖励递减值
MAX_EPISODES = 13  # 最大训练轮数
FRESH_TIME = 0.3  # 移动时间间隔


# 创建Q表
def build_q_table(n_states, actions):
    table = pd.DataFrame(
        np.zeros((n_states, len(actions))),
        columns=actions
    )
    '''
    q_table
       left  right
    0   0.0    0.0
    1   0.0    0.0
    2   0.0    0.0
    3   0.0    0.0
    4   0.0    0.0
    5   0.0    0.0
    '''
    return table


# 选择向前或者向后的动作
def choose_action(state, q_table):
    state_actions = q_table.iloc[state, :]  # 选出当前state的所有动作
    if (np.random.uniform() > EPSILON) or (state_actions.all() == 0):  # 非贪婪 or 这个state 还没探索过
        action_name = np.random.choice(ACTIONS)
    else:
        action_name = state_actions.idxmax()  # 贪婪模式，选择最大值
    return action_name  # 'left' or 'right'


# 输入选择的动作和状态获取环境反馈和奖励
def get_env_feedback(S, A):
    if A == 'right':
        if S == N_STATES - 2:
            S_ = 'terminal'
            R = 1
        else:
            S_ = S + 1
            R = 0
    else:  # move left
        R = 0
        if S == 0:
            S_ = S
        else:
            S_ = S - 1
    return S_, R


# 过程可视化
def update_env(S, episode, step_counter):
    env_list = ['-'] * (N_STATES - 1) + ['T']
    if S == 'terminal':
        interaction = 'Episode %s: total_steps = %s' % (episode + 1, step_counter)
        print('\r{}'.format(interaction), end='')
        time.sleep(2)
        print('\r                      ', end='')
    else:
        env_list[S] = 'O'
        interaction = ''.join(env_list)
        print('\r{}'.format(interaction), end='')
        time.sleep(FRESH_TIME)


def rl():
    q_table = build_q_table(N_STATES, ACTIONS)  # 初始化 q table

    plt.figure()
    plt.ion()
    plt.bar(q_table.index, q_table['left'], facecolor='#9999ff', edgecolor='white')
    plt.bar(q_table.index, q_table['right'], facecolor='#9999ff', edgecolor='white')
    plt.show()

    for episode in range(MAX_EPISODES):  # 回合
        step_counter = 0
        S = 0  # 回合初始位置
        is_terminated = False  # 是否回合结束
        update_env(S, episode, step_counter)
        while not is_terminated:
            A = choose_action(S, q_table)  # 选择当前行为 left or right
            S_, R = get_env_feedback(S, A)  # 根据行为获取当前状态(执行动作后的位置，奖励)
            q_predict = q_table.loc[S, A]  # 当前行为估算的值
            if S_ != 'terminal':
                q_target = R + GAMMA * q_table.iloc[S_, :].max()
            else:
                q_target = R
                is_terminated = True
            q_table.loc[S, A] += ALPHA * (q_target - q_predict)  # 更新q_table 训练核心
            S = S_  # 探索者移动到下一个 state

            update_env(S, episode, step_counter + 1)

            plt.bar(q_table.index, q_table['left'], facecolor='#9999ff', edgecolor='white')
            plt.bar(q_table.index, -q_table['right'], facecolor='#ff9999', edgecolor='white')
            plt.pause(0.1)

            step_counter += 1
    return q_table


if __name__ == '__main__':
    q_table = rl()
    print('\r\nQ-table:\n')
    print(q_table)
