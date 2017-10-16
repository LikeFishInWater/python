import tensorflow as tf

a = tf.constant([1., 2., 3.], shape=[3], name='a')
b = tf.constant([1., 2., 3.], shape=[3], name='a')
c = a + b
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
print(str(sess.run(c)))

# with tf.device('/cpu:0'):
#   a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
#   b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
# c = tf.matmul(a, b)
# sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# print(str(sess.run(c)))

# with tf.device('/gpu:0'):
#     a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
#     b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
#     c = tf.matmul(a, b)
# sess = tf.Session(config=tf.ConfigProto(
#     allow_soft_placement=False, log_device_placement=True))
# print(str(sess.run(c)))
