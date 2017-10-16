import math
import numpy as np
import tensorflow as tf

# # parameter
# w1 = np.ones(32)
# w2 = np.ones(15)
w1 = [0.76710713, 0.60553187, 0.38654989, 0.65817481, 0.78794152,
      0.58775622, 0.77491963, 1.15229785, 0.56947005, 0.63482797,
      0.64538503, 0.68018997, 0.98460859, 0.83970416, 0.72892302,
      0.97554332, 0.55641484, 0.60240734, 0.55333674, 0.68954241,
      0.57166761, 0.7890203, 0.93959165, 1.26261044, 0.7655015,
      0.54993391, 0.52421993, 0.72883743, 0.75430727, 0.63908386,
      0.81096911, 1.0136894]
w2 = [8.7302866, 7.21266699, 6.52828598, 7.82843494, 6.89069176,
      8.23283482, 8.00707245, 7.28282022, 8.41993237, 7.92794991,
      7.86982584, 8.87649536, 7.80749989, 8.64155388, 7.86943626]
EbN0 = 7
# # BCH(15,11,1)
Rate = 11 / 15
G = np.array([[1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1]])
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
EsN0 = EbN0 - 10 * np.log(1 / Rate) / np.log(10)
sigma = math.sqrt(1 / 2 * 10 ** (-EsN0 / 10))
scale = 10000
trans = np.zeros((scale, 15))
for i in range(scale):
    inf = np.random.randint(2, size=11)
    for j in range(11):
        trans[i] += inf[j] * G[j]
    trans[i] %= 2
    trans[i] = trans[i] * 2 - 1
recei = np.ones((scale, 15))
for i in range(scale):
    recei[i] = trans[i] + sigma * np.random.randn(15)
# # graph
x = tf.placeholder(shape=[15], dtype=tf.float32)
y = tf.placeholder(shape=[15], dtype=tf.float32)
ops = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
# w2 = np.ones(15)
output = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
outBit = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# check node
for i in range(4):
    for j in range(8):
        ops_temp = 1
        for t in range(8):
            if t != j:
                ops_temp *= (math.e ** (x[Hr[i][t]] / 2) - math.e ** (-x[Hr[i][t]] / 2)) / (
                    math.e ** (x[Hr[i][t]] / 2) + math.e ** (-x[Hr[i][t]] / 2))
        ops[0][8 * i + j] = 2 * 1 / 2 * tf.log(2 / (1 - ops_temp) - 1)
# variable node
for i in range(4):
    for j in range(8):
        ops_temp = w2[Hr[i][j]] * x[Hr[i][j]]
        for t in range(4):
            if t != i and H[t][Hr[i][j]] == 1:
                ops_temp += w1[8 * t + j] * ops[0][Hi[t][Hr[i][j]]]
                if Hi[t][Hr[i][j]] == -1:
                    print('error')
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
    output[i] = w2[i] * x[i]
for i in range(4):
    for j in range(8):
        output[Hr[i][j]] += w1[8 * i + j] * ops[2][8 * i + j]
for i in range(15):
    outBit[i] = tf.cond(output[i] > 0, lambda: 1., lambda: -1.)
err = tf.reduce_mean(abs(outBit - y) / 2)

# testing
sess = tf.Session()
err_vec = []
for i in range(scale):
    err_temp = sess.run(err, feed_dict={x: recei[i], y: trans[i]})
    err_vec.append(err_temp)
    if i % 10 == 0:
        print(str(sum(err_vec) / len(err_vec)))
sess.close()

# 检验编码正确性
# for i in range(scale):
#     a = np.zeros(4)
#     for j in range(4):
#         a[j] = 0
#         for t in range(15):
#             a[j] += (trans[i][t]+1)/2 * H[j][t]
#     a %= 2
#     print(str(a))
