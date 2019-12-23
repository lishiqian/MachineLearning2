from src.reinforcement_mine_demo.RL02_Sarsa.RL_brain import SarsaTable
from src.reinforcement_mine_demo.RL02_Sarsa.RL_brain import SarasLambdaTable
from src.reinforcement_mine_demo.RL02_Sarsa.maze_env import Maze


def update():
    for i in range(200):
        observation = env.reset()

        action = RL.choose_action(str(observation))

        while True:
            env.render()

            observation_, r, done = env.step(action)

            action_ = RL.choose_action(str(observation_))

            RL.learn(str(observation), action, r, str(observation_), action_)

            if done:
                print("Episode %d complete!" % i)
                break

            observation = observation_
            action = action_


if __name__ == '__main__':
    env = Maze()
    # RL = SarsaTable(actions=list(range(env.n_actions)))
    RL = SarasLambdaTable(actions=list(range(env.n_actions)))

    env.after(100, update())
    env.mainloop()
