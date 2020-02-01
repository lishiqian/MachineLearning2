"""
Deep Deterministic Policy Gradient (DDPG), Reinforcement Learning.
DDPG is Actor Critic based algorithm.
"""

import tensorflow as tf
import numpy as np


class DDPG(object):

    def __init__(self, a_dim, s_dim, a_bound,
                 memory_capacity=10000,
                 batch_size=32,
                 soft_replacement=0.01,
                 reward_decay=0.9,
                 lr_a=0.001,
                 lr_c=0.002,
                 output_graph=False):  # a_bound 动作的范围，Pendulum是[-2, 2]
        self.a_dim = a_dim
        self.s_dim = s_dim
        self.a_bound = a_bound
        self.memory_size = memory_capacity
        self.batch_size = batch_size
        self.TAU = soft_replacement  # 替换神经网络时eval的替换权重
        self.gamma = reward_decay
        self.lr_a = lr_a
        self.lr_c = lr_c

        self.memory = np.zeros((memory_capacity, s_dim * 2 + a_dim + 1))
        self.pointer = 0
        self.sess = tf.Session()

        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')

        with tf.variable_scope('Actor'):
            self.a = self._build_a(self.S, scope='eval', trainable=True)
            a_ = self._build_a(self.S_, scope='target', trainable=False)

        with tf.variable_scope('Critic'):
            q = self._build_c(self.S, self.a, scope='eval', trainable=True)
            q_ = self._build_c(self.S_, a_, scope='target', trainable=False)

        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval')
        self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target')
        self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval')
        self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target')

        # 将eval的参数替换给target时，并不是直接覆盖，而是eval取0.99而原来的target参数保留0.01
        self.soft_replace = [tf.assign(t, (1 - self.TAU) * t + self.TAU * e) for t, e in
                             zip(self.at_params + self.ct_params, self.ae_params + self.ce_params)]

        q_target = self.R + self.gamma * q_
        td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
        self.ctrain = tf.train.AdamOptimizer(self.lr_c).minimize(td_error, var_list=self.ce_params)

        a_loss = - tf.reduce_mean(q)  # maximize the q
        self.atrain = tf.train.AdamOptimizer(self.lr_a).minimize(a_loss, var_list=self.ae_params)

        self.sess.run(tf.global_variables_initializer())
        if output_graph:
            tf.summary.FileWriter('logs/', graph=self.sess.graph)

    def choose_action(self, s):
        return self.sess.run(self.a, {self.S: s[np.newaxis, :]})[0]

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % self.memory_size
        self.memory[index, :] = transition
        self.pointer += 1

    def learn(self):
        self.sess.run(self.soft_replace)

        indices = np.random.choice(self.memory_size, self.batch_size)
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        br = bt[:, -self.s_dim - 1: -self.s_dim]
        bs_ = bt[:, -self.s_dim:]

        self.sess.run(self.atrain, {self.S: bs})
        self.sess.run(self.ctrain, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_})

    def _build_a(self, s, scope, trainable):
        with tf.variable_scope(scope):
            net = tf.layers.dense(s, 30, activation=tf.nn.relu, name='l1', trainable=trainable)
            a = tf.layers.dense(net, self.a_dim, activation=tf.nn.tanh, name='a', trainable=trainable)
            return tf.multiply(a, self.a_bound, name='scaled_a')  # 将a的值得输出范围从(-1, 1)变为(-2, 2) tanh 值域为(-1, +1)

    def _build_c(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            n_l1 = 30
            w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            return tf.layers.dense(net, 1, trainable=trainable)  # Q(s,a)
