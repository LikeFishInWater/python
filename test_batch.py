import tensorflow as tf
import numpy as np

x = tf.placeholder(shape=[5], dtype=tf.float32)
y = tf.placeholder(shape=[], dtype=tf.float32)

train_size = 1000
x_vals = np.random.randn(5, train_size)
# x_vals=np.linspace(0,1,train_size)
y_vals = 1 * x_vals[0] + 2 * x_vals[1] + 3 * x_vals[2] + 4 * x_vals[3] + 5 * x_vals[4] + 10

A = tf.Variable(tf.random_normal(shape=[5]))
B = tf.Variable(tf.random_normal(shape=[]))
temp = 0
for i in range(5):
    temp += A[i] * x[i]
temp += B
model_output = temp

init = tf.global_variables_initializer()
learning_rate = 0.1
my_opt = tf.train.GradientDescentOptimizer(learning_rate)
loss = tf.reduce_mean(tf.square(model_output - y))
train = my_opt.minimize(loss)

sess = tf.Session()
sess.run(init)

print('initial: A=' + str(sess.run(A)) + '  B=' + str(sess.run(B)))
for i in range(train_size):
    sess.run(train, feed_dict={x: x_vals[:, i], y: y_vals[i]})
print('training: A=' + str(sess.run(A)) + '  B=' + str(sess.run(B)))
