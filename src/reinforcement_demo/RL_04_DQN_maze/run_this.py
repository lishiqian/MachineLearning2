from src.reinforcement_demo.RL_04_DQN_maze.maze_env import Maze
from src.reinforcement_demo.RL_04_DQN_maze.RL_brain import DeepQNetwork


def update():
    step = 0  # 记录步骤数
    # 学习 100 回合
    for episode in range(300):
        # 初始化环境 observation [-0.5 -0.5]
        observation = env.reset()

        while True:
            # 更新可视化环境
            env.render()

            # RL 大脑根据 state 的观测值挑选 action
            action = RL.choose_action(observation)

            # 探索者在环境中实施这个 action, 并得到环境返回的下一个 state 观测值, reward 和 done (是否是掉下地狱或者升上天堂)
            # observation_ 格式[-0.5 - 0.5]
            observation_, reward, done = env.step(action)

            # DQN存储记忆
            RL.store_transition(observation, action, reward, observation_)

            # 控制学习的起始时间和频率（先积累一些记忆再学习）
            if (step >= 200) and (step % 5 == 0):
                RL.learn()

            # 将下一个 state 的值传到下一次循环
            observation = observation_

            # 如果掉下地狱或者升上天堂, 这回合就结束了
            if done:
                print("Episode %d complate!" % episode)
                break
            step += 1

    # 结束游戏并关闭窗口
    print('game over')
    RL.plot_cost()
    env.destroy()


if __name__ == "__main__":
    env = Maze()
    RL = DeepQNetwork(env.n_actions,
                      env.n_features,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200,  # 每 200 步替换一次 target_net 的参数
                      memory_size=2000,  # 记忆上限
                      # output_graph=True   # 是否输出 tensorboard 文件
                      )
    env.after(100, update)
    env.mainloop()
    RL.plot_cost()  # 观看神经网络的误差曲线
