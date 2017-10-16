import math
import numpy as np
import tensorflow as tf

# # BCH(15,11,1)
Rate = 11 / 15
H = np.array([[1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0],
              [0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0],
              [0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0],
              [1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1]])
Hr = np.array([[0, 1, 2, 3, 5, 7, 8, 11],
               [1, 2, 3, 4, 6, 8, 9, 12],
               [2, 3, 4, 5, 7, 9, 10, 13],
               [0, 1, 2, 4, 6, 7, 10, 14]])
Hi = np.array([[0, 1, 2, 3, -1, 4, -1, 5, 6, -1, -1, 7, -1, -1, -1],
               [-1, 8, 9, 10, 11, -1, 12, -1, 13, 14, -1, -1, 15, -1, -1],
               [-1, -1, 16, 17, 18, 19, -1, 20, -1, 21, 22, -1, -1, 23, -1],
               [24, 25, 26, -1, 27, -1, 28, 29, -1, -1, 30, -1, -1, -1, 31]])
EsN0 = 3
EbN0 = EsN0 + 10 * np.log(1 / Rate) / np.log(10)
sigma = math.sqrt(1 / 2 * 10 ** (-EsN0 / 10))
scale = 10000
train_scale = 9000
test_scale = 1000
trans = np.ones((scale, 15))
for i in range(scale):
    trans[i] = np.ones(15)
for i in range(test_scale):
    test_inf = np.random.randint(2, size=11)
    trans[train_scale + i] = np.zeros(15)
    for j in range(11):
        trans[train_scale + i] += test_inf[j] * G[j]
    trans[train_scale + i] %= 2
    trans[train_scale + i]=trans[train_scale + i]*2-1
recei = np.ones((scale, 15))
for i in range(scale):
    recei[i] = trans[i] + sigma * np.random.randn(15)
# # graph
x = tf.placeholder(shape=[None, 15], dtype=tf.float32)
y = tf.placeholder(shape=[None, 15], dtype=tf.float32)
ops = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
w1 = tf.Variable(tf.ones(32, dtype=tf.float32))
# w1 = tf.Variable(tf.random_normal(shape=[32]))
w2 = tf.Variable(tf.ones(15, dtype=tf.float32))
# w2 = np.ones(15)
output = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
output1 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
outBit = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

# check node
for i in range(4):
    for j in range(8):
        ops_temp = 1
        for t in range(8):
            if t != j:
                ops_temp *= (math.e ** (x[:, Hr[i][t]] / 2) - math.e ** (-x[:, Hr[i][t]] / 2)) / (
                    math.e ** (x[:, Hr[i][t]] / 2) + math.e ** (-x[:, Hr[i][t]] / 2))
        ops[0][8 * i + j] = 2 * 1 / 2 * tf.log(2 / (1 - ops_temp) - 1)
# variable node
for i in range(4):
    for j in range(8):
        ops_temp = w2[Hr[i][j]] * x[:, Hr[i][j]]
        for t in range(4):
            if t != i and H[t][Hr[i][j]] == 1:
                ops_temp += w1[8 * t + j] * ops[0][Hi[t][Hr[i][j]]]
        ops_temp /= 2
        ops[1][8 * i + j] = (math.e ** ops_temp - math.e ** (-ops_temp)) / (
            math.e ** ops_temp + math.e ** (-ops_temp))
# check node
for i in range(4):
    for j in range(8):
        ops_temp = 1
        for t in range(8):
            if t != j:
                ops_temp *= ops[1][8 * i + t]
        ops[2][8 * i + j] = 2 * 1 / 2 * tf.log(2 / (1 - ops_temp) - 1)
# output layer
for i in range(15):
    output[i] = w2[i] * x[:, i]
for i in range(4):
    for j in range(8):
        output[Hr[i][j]] += w1[8 * i + j] * ops[2][8 * i + j]
cross_temp = 0
for i in range(15):
    output1[i] = 1 / (1 + math.e ** (-output[i]))
    cross_temp += (y[:, i] + 1) / 2 * tf.log(output1[i] + 1e-10) + (1 - (y[:, i] + 1) / 2) * tf.log(
        1 - output1[i] + 1e-10)
cross_entropy = tf.reduce_mean(cross_temp) / (-15)

# #training
learning_rate = 1e-2
# train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
loss_vec = []
batch_size = 100
for i in range(train_scale):
    random_index = np.random.choice(train_scale, size=batch_size)
    T_recei = recei[random_index]
    T_trans = trans[random_index]
    # for j in range(1000):
    sess.run(train_step, feed_dict={x: T_recei, y: T_trans})
    temp_loss = sess.run(cross_entropy, feed_dict={x: T_recei, y: T_trans})
    if i % 100 == 0:
        print('temp_loss=' + str(temp_loss))
        print('w1=' + str(sess.run(w1)) + 'w2=' + str(sess.run(w2)))

# for i in range(15):
#     outBit[i] = tf.cond(output[i] > 0, lambda: 1., lambda: -1.)
# err = tf.reduce_sum(abs(outBit - y) / 2)
errR = []
for i in range(test_scale):
    temp_errR = sess.run(cross_entropy, feed_dict={x: [recei[train_scale + i]], y: [trans[train_scale + i]]})
    errR.append(temp_errR)
    if i % 10 == 0:
        print(str(temp_errR))
print(str(errR))
sess.close()
