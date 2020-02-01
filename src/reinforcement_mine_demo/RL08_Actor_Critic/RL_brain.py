import numpy as np
import tensorflow as tf


class Actor(object):

    def __init__(self, sess, n_features, n_actions, lr=0.001):
        self.sess = sess

        self.s = tf.placeholder(tf.float32, [1, n_features], name='state')
        self.a = tf.placeholder(tf.int32, None, name='action')
        self.td_error = tf.placeholder(tf.float32, None, name='td_error')

        with tf.variable_scope('Actor'):
            l1 = tf.layers.dense(
                inputs=self.s,
                units=20,
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0.0, 0.1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='l1'
            )

            self.acts_prob = tf.layers.dense(
                inputs=l1,
                units=n_actions,
                activation=tf.nn.softmax,
                kernel_initializer=tf.random_normal_initializer(0.0, 0.1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biase
                name='acts_prob'
            )

        with tf.variable_scope('exp_v'):
            log_prob = tf.log(self.acts_prob[0, self.a])
            self.exp_v = tf.reduce_mean(log_prob * self.td_error)  # advantage (TD_error) guided loss

        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(-self.exp_v)  # minimize(-exp_v) = maximize(exp_v)

    def learn(self, s, a, td_error):
        # s, a 用于产生 Gradient ascent 的方向,
        # td_error 来自 Critic, 用于告诉 Actor 这方向对不对.
        s = s[np.newaxis, :]
        feed_dict = {self.s: s, self.a: a, self.td_error: td_error}

        _, exp_v = self.sess.run([self.train_op, self.exp_v], feed_dict)
        return exp_v

    def choose_action(self, s):
        # 根据 s 选 行为 a
        s = s[np.newaxis, :]
        probs = self.sess.run(self.acts_prob, feed_dict={self.s: s})
        return np.random.choice(np.arange(probs.shape[1]), p=probs.ravel())


class Critic(object):

    def __init__(self, sess, n_features, lr=0.01, reward_decay=0.9):
        self.sess = sess
        self.gamma = reward_decay  # 奖励衰减

        self.s = tf.placeholder(tf.float32, [1, n_features], name='state')
        self.v_ = tf.placeholder(tf.float32, [1, 1], name='v_next')  # 下一步状态能取到的估计奖励值
        self.r = tf.placeholder(tf.float32, None, name='reward')  # 从环境中取到的实际奖励值

        with tf.variable_scope('Critic'):
            l1 = tf.layers.dense(
                inputs=self.s,
                units=20,
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0.0, 0.1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='l1'
            )

            self.v = tf.layers.dense(
                inputs=l1,
                units=1,
                activation=None,
                kernel_initializer=tf.random_normal_initializer(0.0, 0.1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='v'
            )

        with tf.variable_scope('squared_TD_error'):
            self.td_error = self.r + self.gamma * self.v_ - self.v
            self.loss = tf.square(self.td_error)  # TD_error = (r+gamma*V_next) - V_eval

        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)

    def learn(self, s, r, s_):
        # 学习 状态的价值 (state value), 不是行为的价值 (action value),
        # 计算 TD_error = (r + v_) - v,
        # 用 TD_error 评判这一步的行为有没有带来比平时更好的结果,
        # 可以把它看做 Advantage
        # 学习时产生的 TD_error
        s, s_ = s[np.newaxis, :], s_[np.newaxis, :]

        v_ = self.sess.run(self.v, feed_dict={self.s: s_})
        td_error, _ = self.sess.run([self.td_error, self.train_op], feed_dict={self.s: s, self.v_: v_, self.r: r})
        return td_error
