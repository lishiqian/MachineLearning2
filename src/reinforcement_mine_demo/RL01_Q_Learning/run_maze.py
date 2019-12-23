from src.reinforcement_mine_demo.RL01_Q_Learning.RL_brain import QLearningTable
from src.reinforcement_mine_demo.RL01_Q_Learning.maze_env import Maze


def update():
    for i in range(100):
        # 重新初始化环境
        observation = env.reset()

        while True:
            env.render()

            action = RL.choose_action(str(observation))  # 选择一个动作

            observation_, R, done = env.step(action)  # 环境根据动作给的下一步状态，奖励，是否结束游戏标志

            RL.learn(str(observation), action, R, str(observation_))  # 学习

            if done:
                print("Episode %d complete！" % i)
                break

            observation = observation_


if __name__ == '__main__':
    env = Maze()
    RL = QLearningTable(actions=list(range(env.n_actions)))

    env.after(100, update)
    env.mainloop()
