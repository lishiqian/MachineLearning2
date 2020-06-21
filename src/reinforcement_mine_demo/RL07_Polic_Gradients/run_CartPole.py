import gym
import matplotlib.pyplot as plt
from src.reinforcement_mine_demo.RL07_Polic_Gradients.RL_brain import PolicGradient

DISPLAY_REWARD_THRESHOLD = 150  # renders environment if total episode reward is greater then this threshold
# DISPLAY_REWARD_THRESHOLD = 0  # renders environment if total episode reward is greater then this threshold
RENDER = False  # rendering wastes time

env = gym.make('CartPole-v0')
env.seed(1)  # reproducible, general Policy gradient has high variance
env = env.unwrapped

print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)

RL = PolicGradient(
    n_actions=env.action_space.n,
    n_features=env.observation_space.shape[0],
    learning_rate=0.02,
    reward_decay=0.98,
    output_graph=False
)

for i_episode in range(3000):

    observation = env.reset()

    while True:
        if RENDER: env.render()

        action = RL.choose_action(observation)

        observation_, reward, done, info = env.step(action)

        RL.store_transition(observation, action, reward)

        if done:
            ep_rs_sum = sum(RL.ep_rs)

            if 'running_reward' not in globals():
                running_reward = ep_rs_sum
            else:
                running_reward = running_reward * 0.99 + ep_rs_sum * 0.01
            if running_reward >= DISPLAY_REWARD_THRESHOLD: RENDER = True

            print('Episode: %d, reward:%d' % (i_episode, int(running_reward)))

            vt = RL.learn()

            if i_episode == 0:
                plt.plot(vt)
                plt.xlabel('episode steps')
                plt.ylabel('normalized state-action value')
                plt.show()
            break

        observation = observation_