import tensorflow as tf
import numpy as np


class DoubleDQN(object):
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
            double_q=True,
            output_graph=False,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay  # 奖励衰减
        self.epsilon_max = e_greedy  # 最大贪婪度
        self.replace_target_iter = replace_target_iter  # 间隔多少步替换 target_net的参数
        self.memory_size = memory_size  # 记忆上限
        self.batch_size = batch_size  # 每次从记忆库中抽取batch_size这么多条数据去训练
        self.epsilon_increment = e_greedy_increment  # 贪婪度增加量
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max  # 贪婪度
        self.double_q = double_q

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

            with tf.variable_scope('l1'):
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

            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s_, w1) + b1)

            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_next = tf.matmul(l1, w2) + b2

    def choose_action(self, observation):
        observation = observation[np.newaxis, :]

        if np.random.uniform() < self.epsilon:  # 贪婪模式
            action_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(action_value)
        else:  # 探索模式
            action = np.random.randint(0, self.n_actions)
        return action

    def store_transition(self, state, action, reward, state_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        transition = np.hstack((state, [action, reward], state_))

        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # 替换target_net参数
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            print('target_params_replace %d' % (self.learn_step_counter / self.replace_target_iter))

        # 随机抽取batch_size条数据进行训练
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        # 这一段和DQN不一样
        q_next, q_eval4next = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={
                self.s: batch_memory[:, -self.n_features:],
                self.s_: batch_memory[:, -self.n_features:]
            }
        )
        q_eval = self.sess.run(self.q_eval, feed_dict={self.s: batch_memory[:, :self.n_features]})

        q_target = q_eval.copy()
        # batch_index = [0,1,2,3...,self.batch_size-1]
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        # 获取当前记忆的动作 eval_act_index = batch_memory[:, 2]
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        # 获取当前记忆的奖励 reward = batch_memory[:, 3]
        reward = batch_memory[:, self.n_features + 1]

        # 这里和DQN也不一样
        # 当前状态选择当前动作得到下一个状态的最大奖励(q_next给出)
        if self.double_q:
            max_act4next = np.argmax(q_eval4next, axis=1)
            selected_q_next = q_next[batch_index, max_act4next]
        else:
            selected_q_next = np.max(q_next, axis=1)

        q_target[batch_index, eval_act_index] = reward + self.gamma * selected_q_next

        _, self.cost = self.sess.run(
            [self._train_op, self.loss],
            feed_dict={
                self.s: batch_memory[:, :self.n_features],
                self.q_target: q_target
            }
        )
        self.cost_his.append(self.cost)

        # 增加贪婪度，降低动作选择的随机性
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.xlabel('train_step')
        plt.ylabel('Cost')
        plt.show()

    def save(self):
        saver = tf.train.Saver()
        saver.save(self.sess, 'net/params', write_meta_graph=False)

    def restore(self):
        saver = tf.train.Saver()
        saver.restore(self.sess, 'net/params')
