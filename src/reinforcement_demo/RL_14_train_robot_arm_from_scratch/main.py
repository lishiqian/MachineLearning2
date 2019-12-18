from src.reinforcement_demo.RL_14_train_robot_arm_from_scratch.env import ArmEnv
from src.reinforcement_demo.RL_14_train_robot_arm_from_scratch.rl import DDPG

# # 设置全局变量
# MAX_EPISODES = 500
# MAX_EP_STEPS = 200
#
# # 设置环境
# env = ArmEnv()
# s_dim = env.state_dim
# a_dim = env.action_dim
# a_bound = env.action_bound
#
# # 设置学习方法 (这里使用 DDPG)
# rl = DDPG(a_dim, s_dim, a_bound)
#
# # 开始训练
# for i in range(MAX_EPISODES):
#     s = env.reset()                 # 初始化回合设置
#     for j in range(MAX_EP_STEPS):
#         env.render()                # 环境的渲染
#         a = rl.choose_action(s)     # RL 选择动作
#         s_, r, done = env.step(a)   # 在环境中施加动作
#
#         # DDPG 这种强化学习需要存放记忆库
#         rl.store_transition(s, a, r, s_)
#
#         if rl.memory_full:
#             rl.learn()              # 记忆库满了, 开始学习
#
#         s = s_                      # 变为下一回合


MAX_EPISODES = 500
MAX_EP_STEPS = 300
ON_TRAIN = False

# set env
env = ArmEnv()
s_dim = env.state_dim
a_dim = env.action_dim
a_bound = env.action_bound

# set RL method (continuous)
rl = DDPG(a_dim, s_dim, a_bound)


def train():
    # start training
    for i in range(MAX_EPISODES):
        s = env.reset()
        ep_r = 0.
        for j in range(MAX_EP_STEPS):
            #env.render()

            a = rl.choose_action(s)

            s_, r, done = env.step(a)

            rl.store_transition(s, a, r, s_)

            ep_r += r
            if rl.memory_full:
                # start to learn once has fulfilled the memory
                rl.learn()

            s = s_
            if done or j == MAX_EP_STEPS-1:
                print('Ep: %i | %s | ep_r: %.1f | steps: %i' % (i, '---' if not done else 'done', ep_r, j))
                break
    rl.save()


def eval():
    rl.restore()
    env.render()
    env.viewer.set_vsync(True)
    while True:
        s = env.reset()
        for _ in range(200):
            env.render()
            a = rl.choose_action(s)
            s, r, done = env.step(a)
            if done:
                break


if ON_TRAIN:
    train()
else:
    eval()