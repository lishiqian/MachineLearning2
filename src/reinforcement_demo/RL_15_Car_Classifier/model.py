import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from src.reinforcement_demo.RL_15_Car_Classifier.data_processing import load_data
from src.reinforcement_demo.RL_15_Car_Classifier.data_processing import convert2onehot

# 加载数据
data = load_data()
new_data = convert2onehot(data)
new_data = new_data.values.astype(np.float32)
np.random.shuffle(new_data)  # 打乱数据顺序

# 将数据划分成训练集合测试集
sep = int(0.7 * len(data))
training_data = new_data[:sep]
test_data = new_data[sep:]

# 定义神经网络
tf_input = tf.placeholder(np.float32, [None, 25], name='tf_input')
tfx = tf_input[:, :21]
tfy = tf_input[:, 21:]

l1 = tf.layers.dense(tfx, 128, tf.nn.relu, name='l1')
l2 = tf.layers.dense(l1, 128, tf.nn.relu, name='l2')
out = tf.layers.dense(l1, 4, name='out')
predict = tf.nn.softmax(out, name='pre')

loss = tf.losses.softmax_cross_entropy(onehot_labels=tfy, logits=out)
train_op = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
accuracy = tf.metrics.accuracy(labels=tf.argmax(tfy, axis=1), predictions=tf.argmax(predict, axis=1))[1]

sess = tf.Session()
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())

accuracy_list = []
plt.ion()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
for i in range(4000):
    batch_index = np.random.randint(len(training_data), size=32)
    sess.run(train_op, {tf_input: training_data[batch_index]})

    if i % 50 == 0:
        loss_, accuracy_, pre_ = sess.run([loss, accuracy, predict], {tf_input: test_data})
        print('step=', i, 'loss=', loss_, 'accuracy=', accuracy_)

        ax1.cla()
        pre_sum = [0, 0, 0, 0]
        for p in pre_:
            pre_sum[np.argmax(p)] += 1

        ax1.bar(np.arange(0, 8, 2), pre_sum)
        ax1.bar(np.arange(0, 8, 2) + 0.5, np.sum(test_data[:, 21:], axis=0))

        ax2.cla()
        accuracy_list.append(accuracy_)
        ax2.plot(np.arange(len(accuracy_list)), accuracy_list)

        plt.pause(0.2)

plt.ioff()
plt.show()
