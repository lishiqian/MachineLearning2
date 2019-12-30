from src.reinforcement_mine_demo.RL03_DQN.maze_env import Maze
from src.reinforcement_mine_demo.RL03_DQN.RL_brain import DeepQNetwork


# from src.reinforcement_demo.RL_04_DQN_maze.RL_brain import DeepQNetwork


def train():
    step = 0
    for episode in range(300):
        # 初始化环境
        observation = env.reset()

        while True:
            env.render()

            action = RL.choose_action(observation)

            observation_, reward, done = env.step(action)

            RL.store_transition(observation, action, reward, observation_)

            # 控制学习的起始时间和频率（先积累一些记忆再学习）
            if (step >= 200) and (step % 5 == 0):
                RL.learn()

            observation = observation_

            if done:
                print("Episode %d complete!" % episode)
                break

            step += 1

    RL.save()
    # 结束游戏并关闭窗口
    print('game over')
    RL.plot_cost()
    env.destroy()


def run():
    RL.restore()
    while True:
        # 初始化环境
        observation = env.reset()

        while True:
            env.render()

            action = RL.choose_action(observation)

            observation_, reward, done = env.step(action)

            observation = observation_

            if done:
                print("Episode %d complete!")
                break


if __name__ == '__main__':
    env = Maze()
    RL = DeepQNetwork(
        n_actions=env.n_actions,
        n_features=env.n_features,
        learning_rate=0.01,
        e_greedy=0.9,
        replace_target_iter=200,
        memory_size=2000,
        output_graph=False)


    is_train = True
    if is_train:
        env.after(100, train)
    else:
        env.after(100, run)
    env.mainloop()
