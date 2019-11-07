from src.reinforcement_demo.RL_03_Sarsa_maze.maze_env import Maze
from src.reinforcement_demo.RL_03_Sarsa_maze.RL_brain import SarsaTable
from src.reinforcement_demo.RL_03_Sarsa_maze.RL_brain import SarsaLambdaTbTable


def update():
    # 学习 100 回合
    for episode in range(100):
        # 初始化 state 的观测值
        observation = env.reset()

        # Sarsa 根据state 观测选择行为
        action = RL.choose_action(str(observation))

        while True:
            # 更新可视化环境
            env.render()

            # 探索者在环境中实施这个 action, 并得到环境返回的下一个 state 观测值, reward 和 done (是否是掉下地狱或者升上天堂)
            observation_, reward, done = env.step(action)

            # 根据下一个statestr(observation_) 选取下一个 action_
            action_ = RL.choose_action(str(observation_))

            # RL 从这个序列 (state, action, reward, state_) 中学习
            RL.learn(str(observation), action, reward, str(observation_), action_)

            # 将下一个 state 的值传到下一次循环
            observation = observation_
            action = action_

            # 如果掉下地狱或者升上天堂, 这回合就结束了
            if done:
                print("Episode %d complate!" % episode)
                break

    # 结束游戏并关闭窗口
    print('game over')
    print(RL.q_table)
    env.destroy()


if __name__ == "__main__":
    # 定义环境 env 和 RL 方式
    env = Maze()
    RL = SarsaTable(actions=list(range(env.n_actions)), e_greedy=0.8)
    # RL = SarsaLambdaTbTable(actions=list(range(env.n_actions)), e_greedy=0.7)

    # 开始可视化环境 env
    env.after(100, update)
    env.mainloop()
