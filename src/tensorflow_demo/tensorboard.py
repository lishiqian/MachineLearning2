import tensorflow as tf
import numpy as np


# tensorboard --logdir=‘logs/’
def add_layer(intputs, in_size, out_size, layer_name, activation_func=None):
    with tf.name_scope("layer"):
        with tf.name_scope("weight"):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]), name='W')
            tf.summary.histogram(layer_name + '/Weight', Weights)
        with tf.name_scope("biases"):
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, name='b')
            tf.summary.histogram(layer_name + '/biases', biases)
        with tf.name_scope("Wx_plus_b"):
            Wx_plus_b = tf.matmul(intputs, Weights) + biases
    if activation_func is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_func(Wx_plus_b)
    tf.summary.histogram(layer_name + '/outputs', outputs)
    return outputs


x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

with tf.name_scope("input"):
    xs = tf.placeholder(tf.float32, [None, 1], name='x_input')
    ys = tf.placeholder(tf.float32, [None, 1], name='y_input')

l1 = add_layer(xs, 1, 10, "layer_1", activation_func=tf.nn.relu)
l2 = add_layer(l1, 10, 10, "layer_2", activation_func=tf.nn.relu)
predction = add_layer(l2, 10, 1, "layer_3", activation_func=None)

with tf.name_scope("loss"):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - predction), reduction_indices=[1]))
    tf.summary.scalar("loss", loss)
with tf.name_scope("train"):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.global_variables_initializer()

sess = tf.Session()
merged = tf.summary.merge_all()
write = tf.summary.FileWriter('logs/', sess.graph)
sess.run(init)

for i in range(1000):
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    if i % 50 == 0:
        result = sess.run(merged, feed_dict={xs: x_data, ys: y_data})
        write.add_summary(result, i)
