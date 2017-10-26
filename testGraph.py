import tensorflow as tf
import numpy as np

x_vals = np.random.randn(10000, 10)
y_vals = np.zeros((10000, 1))
for i in range(10000):
    for j in range(10):
        y_vals[i] += x_vals[i][j] * (j + 1)

x = tf.placeholder(shape=[None, 10], dtype=tf.float32)
y = tf.placeholder(shape=[None, 1], dtype=tf.float32)
w = tf.Variable(tf.ones(10))
output = 0
for i in range(10):
    # if i % 2 == 0:
    #     output += w[i] * x[:, i]
    # else:
    #     output += x[:, i] / w[i]
    # output += w[i] * x[:, i]
    # output += (np.e**(w[i] * x[:, i])-np.e**(-w[i] * x[:, i]))/(np.e**(w[i] * x[:, i])+np.e**(-w[i] * x[:, i]))
loss = tf.reduce_mean(tf.square(output - y[:, 0]))
tf.summary.scalar('loss', loss)
sess = tf.Session()
learning_rate = 0.01
my_opt = tf.train.AdamOptimizer(learning_rate)
train = my_opt.minimize(loss)
init = tf.global_variables_initializer()
sess.run(init)

merged = tf.summary.merge_all()
writer = tf.summary.FileWriter('TB_Graph/', tf.get_default_graph())
loss_vec = []
batch_size = 10
for i in range(10000):
    random_index = np.random.choice(10000, size=batch_size)
    random_x = x_vals[random_index, :]
    random_y = y_vals[random_index, :]
    sess.run(train, feed_dict={x: random_x, y: random_y})
    summary = sess.run(merged, feed_dict={x: random_x, y: random_y})
    writer.add_summary(summary, i)
    loss_temp = sess.run(loss, feed_dict={x: random_x, y: random_y})
    loss_vec.append(loss_temp)
sess.close()
