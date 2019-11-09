import gym
import time

env = gym.make('CartPole-v0')  # 定义使用 gym 库中的那一个环境


count = 0
while True:
    env.reset()
    for i in range(1000):
        action = env.action_space.sample()
        observation, reward, done, info = env.step(env.action_space.sample())
        if done:
            break
        env.render()
        count += 1
        time.sleep(0.2)

print(count)
