import tensorflow as tf

stat = tf.Variable(1)  # 定义一个变量
cons = tf.constant(1)  # 定义一个常量

new_value = tf.add(stat, cons)
update = tf.assign(stat, new_value)

init = tf.global_variables_initializer()  # 如果有变量定义必须执行这个方法初始化变量

with tf.Session() as sess:
    sess.run(init)
    print(sess.run(stat))
    for _ in range(3):
        sess.run(update)
        print(sess.run(stat))
