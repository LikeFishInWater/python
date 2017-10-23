# 最优化训练，没有输入
import tensorflow as tf

a = 2
b = 1
c = 3
x = tf.Variable(10, dtype=tf.float32)
y = a * x ** 2 + b * x + c

sess = tf.Session()
learning_rate = 1e-2
train_step = tf.train.AdamOptimizer(learning_rate).minimize(y)
init = tf.global_variables_initializer()
sess.run(init)


loss_vec = []
for i in range(10000):
    sess.run(train_step)
    loss = sess.run(y)
    loss_vec.append(loss)
