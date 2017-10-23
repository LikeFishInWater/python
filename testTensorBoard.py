import tensorflow as tf
import numpy as np

x_vals = np.random.randn(1000, 10)
y_vals = np.zeros((1000,10))
for i in range(10):
    y_vals[:][i] += i * x_vals[:][i]

with tf.variable_scope("input"):
    x = tf.placeholder(shape=[10], dtype=tf.float32,name="x")
    w = tf.Variable(tf.ones(10, dtype=tf.float32),name="w")
with tf.variable_scope("first"):
    y = 1
    for i in range(10):
        y = y + w[i] * x[i]

sess=tf.Session()
writer = tf.summary.FileWriter("myDir",sess.graph)
sess.close()