import tensorflow as tf
import numpy as np

H = [[1, 1, 1, 0, 0, 0],
     [1, 0, 0, 1, 1, 0],
     [0, 1, 0, 1, 0, 1],
     [0, 0, 1, 0, 1, 1]]
Hr = [[0, 1, 2], [0, 3, 4], [1, 3, 5], [2, 4, 5]]
Hc = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]
Hi = [[0, 1, 2, -1, -1, -1],
      [3, -1, -1, 4, 5, -1],
      [-1, 6, -1, 7, -1, 8],
      [-1, -1, 9, -1, 10, 11]]
H_re = [[[-0.6351, 0.1392, 0, 0], [0.7851, 0, -0.7851, 0], [-0.0055, 0, 0, -0.6351], [0, 0.7851, 0.1392, 0],
         [0, -0.1815, 0, 0.7851], [0, 0, 0.7851, -0.0055]],
        [[0.1815, 0.4873, 0, 0], [-0.2243, 0, -0.6351, 0], [-0.0193, 0, 0, 0.1815], [0, -0.2243, 0.4873, 0],
         [0, -0.6351, 0, -0.2243], [0, 0, -0.2243, -0.0193]],
        [[-0.1815, -0.4873, 0, 0], [0.2242, 0, 0.6351, 0], [0.0193, 0, 0, -0.1815], [0, 0.2243, -0.4873, 0],
         [0, 0.6351, 0, 0.2243], [0, 0, 0.2243, 0.0193]],
        [[0.6351, -0.1392, 0, 0], [-0.7851, 0, 0.1815, 0], [0.0055, 0, 0, 0.6351], [0, -0.7851, -0.1392, 0],
         [0, 0.1815, 0, -0.7851], [0, 0, -0.7851, 0.0055]]]
H_im = [[[0.4615, -0.1759, 0, 0], [0, 0, -0.1318, 0], [-0.2242, 0, 0, 0.4615], [0, 0, -0.1759, 0], [0, -0.1318, 0, 0],
         [0, 0, 0, -0.2242]],
        [[-0.1318, -0.6156, 0, 0], [0, 0, -0.4615, 0], [-0.7848, 0, 0, -0.1318], [0, 0, -0.6156, 0],
         [0, -0.4615, 0, -0], [0, 0, 0, -0.7848]],
        [[0.1318, 0.6156, 0, 0], [0, 0, 0.4615, 0], [0.7848, 0, 0, 0.1318], [0, 0, 0.6156, 0], [0, 0.4615, 0, 0],
         [0, 0, 0, 0.7848]],
        [[-0.4615, 0.1759, 0, 0], [0, 0, 0.1318, 0], [0.2242, 0, 0, -0.4615], [0, 0, 0.1759, 0], [0, 0.1318, 0, 0],
         [0, 0, 0, 0.2242]]]

scale = 10000
EsN0 = 10
EbN0 = EsN0 - 10 * np.log10(12 / 4)
sigma = np.sqrt(1 / 2 * 10 ** (-EsN0 / 10))
iBit = np.random.randint(0, 4, [scale, 6])
iBitProb = np.zeros((scale, 6, 4))
for i in range(scale):
    for j in range(6):
        iBitProb[i][j][iBit[i][j]] = 1
trans_re = np.zeros((scale, 4))
trans_im = np.zeros((scale, 4))
for i in range(scale):
    for j in range(6):
        trans_re[i] += H_re[iBit[i][j]][j]
        trans_im[i] += H_im[iBit[i][j]][j]
recei_re = trans_re + sigma * np.random.randn(scale, 4)
recei_im = trans_im + sigma * np.random.randn(scale, 4)

x_re = tf.placeholder(shape=[None, 4], dtype=tf.float32)
x_im = tf.placeholder(shape=[None, 4], dtype=tf.float32)
y = tf.placeholder(shape=[None, 6, 4], dtype=tf.float32)
w = tf.Variable(tf.ones(12, dtype=tf.float32))
# b = tf.Variable(tf.ones(12, dtype=tf.float32))
opsT = [[0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25],
        [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25],
        [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25],
        [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25]]
ops = [
    [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
    [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
    [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
]
output = [[0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0]]
# first layer: 1st check node
for i in range(4):
    for j in range(3):
        if j == 0:
            a1 = 1
            a2 = 2
        elif j == 1:
            a1 = 0
            a2 = 2
        else:
            a1 = 0
            a2 = 1
        for t in range(4):
            ops_temp = 0
            for m in range(4):
                for n in range(4):
                    ops_temp += opsT[m][Hi[i][Hr[i][a1]]] * opsT[n][Hi[i][Hr[i][a2]]] * 1 / np.sqrt(
                        np.pi * 2) / sigma * np.e ** (-((x_re[:, i] - H_re[t][Hr[i][j]][i] - H_re[m][Hr[i][a1]][i] -
                                                        H_re[n][Hr[i][a2]][i]) ** 2 + (
                                                           x_im[:, i] - H_im[t][Hr[i][j]][i] - H_im[m][Hr[i][a1]][i] -
                                                           H_im[n][Hr[i][a2]][i]) ** 2) / (2 * sigma ** 2))
                    # opsT[m][Hi[i][Hr[i][a1]]] opsT[n][Hi[i][Hr[i][a2]]]
            ops[0][t][Hi[i][Hr[i][j]]] = ops_temp
for i in range(12):
    for j in range(4):
        ops[0][j][i] *= w[i]
# second layer: function node
for i in range(6):
    for j in range(2):  # Hi[Hc[i][1-j]][i]
        sumT = ops[0][0][Hi[Hc[i][1 - j]][i]] + ops[0][1][Hi[Hc[i][1 - j]][i]] + ops[0][2][Hi[Hc[i][1 - j]][i]] + ops[0][3][Hi[Hc[i][1 - j]][i]]
        for t in range(4):
            ops[0][t][Hi[Hc[i][1 - j]][i]] /= (sumT+1e-50)
            ops[1][t][Hi[Hc[i][j]][i]] = ops[0][t][Hi[Hc[i][1 - j]][i]]
# third layer: check node
for i in range(4):
    for j in range(3):
        if j == 0:
            a1 = 1
            a2 = 2
        elif j == 1:
            a1 = 0
            a2 = 2
        else:
            a1 = 0
            a2 = 1
        for t in range(4):
            ops_temp = 0
            for m in range(4):
                for n in range(4):
                    ops_temp += ops[1][m][Hi[i][Hr[i][a1]]] * ops[1][n][Hi[i][Hr[i][a2]]] * 1 / np.sqrt(
                        np.pi * 2) / sigma * np.e ** (
                        -((x_re[:, i] - H_re[t][Hr[i][j]][i] - H_re[m][Hr[i][a1]][i] - H_re[n][Hr[i][a2]][i]) ** 2 + (
                            x_im[:, i] - H_im[t][Hr[i][j]][i] - H_im[m][Hr[i][a1]][i] - H_im[n][Hr[i][a2]][i]) ** 2) / (
                            2 * sigma ** 2))
            ops[2][t][Hi[i][Hr[i][j]]] = ops_temp
for i in range(12):
    for j in range(4):
        ops[2][j][i] *= w[i]
# output layer: function node
for i in range(6):
    sumT = 0
    for j in range(4):
        output[j][i] = ops[2][j][Hi[Hc[i][0]][i]] * ops[2][j][Hi[Hc[i][1]][i]]
        sumT += output[j][i]
    for j in range(4):
        output[j][i] /= sumT

# cross entropy
cross_temp = 0
for i in range(6):
    cross_temp += y[:, i, 0] * tf.log(output[0][i]) + y[:, i, 1] * tf.log(output[1][i]) + y[:, i, 2] * tf.log(
        output[2][i]) + y[:, i, 3] * tf.log(output[3][i])
cross_entropy = tf.reduce_mean(cross_temp) / (-6)
# # training # #
learning_rate = 1e-3
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
loss_vec = []
batch_size = 1000
f1 = open('SCMA_test.txt', 'w')
for i in range(scale):
    random_index = np.random.choice(scale, size=batch_size)
    T_recei_re = recei_re[random_index]
    T_recei_im = recei_im[random_index]
    T_trans = iBitProb[random_index]
    # sess.run(ops[0], feed_dict={x_re: T_recei_re, x_im: T_recei_im, y: T_trans})
    # for j in range(1000):
    sess.run(train_step, feed_dict={x_re: T_recei_re, x_im: T_recei_im, y: T_trans})
    temp_loss = sess.run(cross_entropy, feed_dict={x_re: T_recei_re, x_im: T_recei_im, y: T_trans})
    loss_vec.append(temp_loss)
    if i % 50 == 0:
        print('temp_loss=' + str(temp_loss))
        f1.write(str(temp_loss) + '\n')
        # print('w=' + str(sess.run(w)))
        f1.write('w=' + str(sess.run(w)))
f1.close()
sess.close()
