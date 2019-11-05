import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def add_layer(intputs, in_size, out_size, activation_func=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(intputs, Weights) + biases
    if activation_func is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_func(Wx_plus_b)
    return outputs


x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise
# y_data = np.square(x_data) - 0.5

xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])

l1 = add_layer(xs, 1, 10, activation_func=tf.nn.relu)
l2 = add_layer(l1, 10, 10, activation_func=tf.nn.relu)
predction = add_layer(l2, 10, 1, activation_func=None)

loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - predction), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
init = tf.global_variables_initializer()

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(x_data, y_data)
plt.ion()
plt.xlim(-1.2, 1.2)
plt.ylim(-1, 1)
plt.show(block=False)

with tf.Session() as sess:
    sess.run(init)
    for i in range(3000):
        sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
        if i % 10 == 0:
            try:
                ax.lines.remove(lines[0])
            except Exception:
                pass
            print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
            predction_value = sess.run(predction, {xs: x_data})
            lines = ax.plot(x_data, predction_value, 'r-', lw=5)
            plt.pause(0.1)
