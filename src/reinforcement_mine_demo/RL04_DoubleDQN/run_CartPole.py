import gym
from src.reinforcement_mine_demo.RL04_DoubleDQN.RL_brain import DoubleDQN

def train():
    total_step = 0

    for episode in range(150):
        # 初始化环境
        observation = env.reset()

        ep_r = 0

        while True:
            env.render()

            action = RL.choose_action(observation)

            observation_, reward, done,info = env.step(action)

            x, x_dot, theta, theta_dot = observation_  # 细分开, 为了修改原配的 reward

            # x 是车的水平位移, 所以 r1 是车越偏离中心, 分越少
            # theta 是棒子离垂直的角度, 角度越大, 越不垂直. 所以 r2 是棒越垂直, 分越高

            x, x_dot, theta, theta_dot = observation_
            r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
            r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
            reward = r1 + r2  # 总 reward 是 r1 和 r2 的结合, 既考虑位置, 也考虑角度, 这样 DQN 学习更有效率

            RL.store_transition(observation, action, reward, observation_)

            # 控制学习的起始时间和频率（先积累一些记忆再学习）
            if total_step >= 1000:
                RL.learn()

            ep_r += reward

            observation = observation_

            if done:
                print('episode: ', episode,
                      'ep_r: ', round(ep_r, 2),
                      ' epsilon: ', round(RL.epsilon, 2))
                break

            total_step += 1

    RL.save()
    # 结束游戏并关闭窗口
    print('game over')
    RL.plot_cost()



def run():
    RL.restore()
    while True:
        # 初始化环境
        observation = env.reset()

        while True:
            env.render()

            action = RL.choose_action(observation)

            observation_, reward, done, info = env.step(action)

            observation = observation_

            if done:
                print("Episode complete!")
                break


if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    env = env.unwrapped
    RL = DoubleDQN(
        n_actions=env.action_space.n,
        n_features=env.observation_space.shape[0],
        learning_rate=0.01,
        e_greedy=0.9,
        replace_target_iter=200,
        memory_size=2000,
        e_greedy_increment=0.0008,
        output_graph=False)


    is_train = False
    if is_train:
        train()
    else:
        RL.epsilon = RL.epsilon_max
        run()