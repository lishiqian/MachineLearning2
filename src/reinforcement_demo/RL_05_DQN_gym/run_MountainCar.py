import gym
from src.reinforcement_demo.RL_05_DQN_CartPole.RL_brain import DeepQNetwork

env = gym.make('MountainCar-v0')  # 定义使用 gym 库中的那一个环境
env = env.unwrapped  # 不做这个会有很多限制

print(env.action_space)  # 查看这个环境中可用的 action 有多少个
print(env.action_space.n)  # 查看这个环境中可用的 action 有多少个
print(env.observation_space)  # 查看这个环境中可用的 state 的 observation 有多少个
print(env.observation_space.shape[0])  # 查看这个环境中可用的 state 的 observation 有多少个
print(env.observation_space.high)  # 查看 observation 最高取值
print(env.observation_space.low)  # 查看 observation 最低取值

RL = DeepQNetwork(
    n_actions=env.action_space.n,
    n_features=env.observation_space.shape[0],
    learning_rate=0.01,
    e_greedy=0.9,
    replace_target_iter=100,
    memory_size=2000,
    e_greedy_increment=0.0008
)

total_steps = 0

for i_episode in range(10):
    observation = env.reset()
    ep_r = 0

    while True:
        env.render()  # 刷新环境

        action = RL.choose_action(observation)

        # print('action:', action)
        observation_, reward, done, info = env.step(action)  # 获取下一个 state

        position, velocity = observation_

        reward = abs(position - (-0.5))

        # 保存这一组记忆
        RL.store_transition(observation, action, reward, observation_)

        if total_steps > 1000:
            RL.learn()  # 学习

        ep_r += reward
        if done:
            get = '| Get' if observation_[0] >= env.unwrapped.goal_position else '| ----'
            print('Epi: ', i_episode,
                  get,
                  '| Ep_r: ', round(ep_r, 4),
                  '| Epsilon: ', round(RL.epsilon, 2))
            break

        observation = observation_
        total_steps += 1
    # 最后输出 cost 曲线
    RL.plot_cost()
