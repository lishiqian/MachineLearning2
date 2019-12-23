import tensorflow as tf
import numpy as np


class DeepQNetwork(object):
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=300,
            memory_size=500,
            batch_size=32,
            e_greedy_increment=None,
            output_graph=False
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay  # 奖励衰减
        self.epsilon_max = e_greedy  # 最大贪婪度
        self.replace_target_iter = replace_target_iter  # 间隔多少步替换 target_net的参数
        self.memory_size = memory_size  # 记忆上限
        self.batch_size = batch_size  # 每次从记忆库中抽取batch_size这么多条数据去训练
        self.e_greedy_increment = e_greedy_increment  # 贪婪度增加量
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max  # 贪婪度

        # 记录训练次数用于更新target_net参数所用
        self.learn_step_counter = 0

        self.memory = np.zeros((self.memory_size, self.n_features * 2 + 2))

        self._build_net()
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        # 将eval_net最新参数更新到target_net参数
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.sess = tf.Session()
        if output_graph:
            tf.summary.FileWriter('logs/', self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        self.cost_his = []  # 记录cost变化，最后plot出来观看

    def choose_action(self, observation):
        pass

    def store_transition(self, state, action, reward, state_):
        pass

    def learn(self):
        pass

    def plot_cost(self):
        pass

    # 创建神经网络
    def _build_net(self):
        # --------------------创建eval_net-------------------------------
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')

        with tf.variable_scope('eval_net'):
            c_names = ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
            n_l1 = 10
            w_initializer = tf.random_normal_initializer(0.0, 0.3)
            b_initializer = tf.constant_initializer(0.1)

            while tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)

            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_eval = tf.matmul(l1, w2) + b2

            with tf.variable_scope('loss'):
                self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
            with tf.variable_scope('train'):
                self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        # --------------------创建target_net-------------------------------
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')
        with tf.variable_scope('target_net'):
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
            n_l1 = 10
            w_initializer = tf.random_normal_initializer(0.0, 0.3)
            b_initializer = tf.constant_initializer(0.1)

            while tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s_, w1) + b1)

            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_next = tf.matmul(l1, w2) + b2
