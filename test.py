import tensorflow as tf
import numpy as np

x = tf.placeholder(shape=[None, 2], dtype=tf.float32)
y = tf.placeholder(shape=[None, 1], dtype=tf.float32)
train_size = 10000
# x_vals=np.random.randn(train_size)
x1_vals = np.random.randn(train_size)
x2_vals = np.random.randn(train_size)
y_vals = 3 * x1_vals + 2 * x2_vals + 4
# A1 = tf.Variable(tf.random_normal(shape=[]))
# A2 = tf.Variable(tf.random_normal(shape=[]))
# B = tf.Variable(tf.random_normal(shape=[]))
A1 = tf.Variable(1.)
A2 = tf.Variable(1.)
B = tf.Variable(1.)
mid = [0, 0]
for i in [0, 1]:
    if i == 0:
        mid[0] = A1 * x[:, 0]
    else:
        mid[1] = A2 * x[:, 1]
model_output = mid[0] + mid[1]+B
# mid0 = A1 * x[:, 0]
# mid1 = A2 * x[:, 1]
# model_output = mid0 + mid1 + B
learning_rate = 0.001
# my_opt = tf.train.GradientDescentOptimizer(learning_rate)
my_opt = tf.train.AdamOptimizer(learning_rate)
loss = tf.reduce_mean(tf.square(model_output - y[:, 0]))
train = my_opt.minimize(loss)
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
print('initial: A1=' + str(sess.run(A1)) + ' A2=' + str(sess.run(A2)) + '  B=' + str(sess.run(B)))
batch_size = 1
random_x = np.ones((batch_size, 2))
loss_vec = []
for i in range(train_size):
    random_index = np.random.choice(train_size, size=batch_size)
    random_x1 = np.transpose(x1_vals[random_index])
    random_x2 = np.transpose(x2_vals[random_index])
    random_x[:, 0] = random_x1
    random_x[:, 1] = random_x2
    random_y = np.transpose([y_vals[random_index]])
    sess.run(train, feed_dict={x: random_x, y: random_y})
    temp_loss = sess.run(loss, feed_dict={x: random_x, y: random_y})
    if i % 100 == 0:
        loss_vec.append(temp_loss)
print('training: A1=' + str(sess.run(A1)) + ' A2=' + str(sess.run(A2)) + '  B=' + str(sess.run(B)))
print(str(loss_vec))
