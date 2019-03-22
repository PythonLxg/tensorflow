# -*- coding utf-8 -*-
# C:\Users\lxg\Documents\Python
# Author:李小根
# Time:2019/2/20
# Coding:utf-8
# 两层简单神经网络（全连接）

import tensorflow as tf
import numpy as np

batch_size = 8

# 产生随机数
dataset_size = 32
X = np.random.rand(dataset_size, 2)  # 输入数据
Y = [[int(x0 + x1 < 1)] for (x0, x1) in X]  # 输入数据的标签

# 定义前向传播过程
# 定义输入，参数和输出
x = tf.placeholder(tf.float32, shape=(None, 2))
y_ = tf.placeholder(tf.float32, shape=(None, 1))

w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

# 定义反向传播过程（损失函数和优化方法）
loss = tf.reduce_mean(tf.square(y - y_))
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
# train_step = tf.train.MomentumOptimizer(0.001, 0.9).minimize(loss)
# train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

# 调用会话执行计算结果
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    # 输出未训练到参数
    print('w1:\n', sess.run(w1))
    print('w1:\n', sess.run(w2))

    # 训练模型
    STEPS = 3000
    for i in range(STEPS):
        start = (i * batch_size) % dataset_size
        end = start + batch_size
        sess.run(train_step, feed_dict={x: X[start: end], y_: Y[start: end]})
        if i % 100 == 0:
            total_loss = sess.run(loss, feed_dict={x: X, y_: Y})
            print('after %d step of loss is %g' % (i, total_loss))

    # 训练后的参数
    print('w1:\n', sess.run(w1))
    print('w2:\n', sess.run(w2))
