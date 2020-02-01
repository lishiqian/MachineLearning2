import gym
import time
import numpy as np
from src.reinforcement_mine_demo.RL09_DDPG.RL_brain import DDPG

MAX_EPISODES = 200
MAX_EP_STEPS = 200

env = gym.make('Pendulum-v0')
env = env.unwrapped
env.seed(1)

s_dim = env.observation_space.shape[0]
a_dim = env.action_space.shape[0]
a_bound = env.action_space.high

ddpg = DDPG(s_dim=s_dim, a_dim=a_dim, a_bound=a_bound, output_graph=False)
RENDER = False

var = 3
t1 = time.time()

for i in range(MAX_EPISODES):
    s = env.reset()
    ep_reward = 0

    for j in range(MAX_EP_STEPS):
        if RENDER:
            env.render()

        a = ddpg.choose_action(s)
        a = np.clip(np.random.normal(a, var), -2, 2)
        s_, r, done, info = env.step(a)
        ddpg.store_transition(s, a, r / 10, s_)

        if ddpg.pointer > ddpg.memory_size:
            var *= 0.9995
            ddpg.learn()

        s = s_
        ep_reward += r

        if j == MAX_EP_STEPS - 1:
            print('Episode:', i, ' Reward: %i' % int(ep_reward), 'Explore: %.2f' % var, )
            if ep_reward > -300: RENDER = True
            break

print('Running time: ', time.time() - t1)
