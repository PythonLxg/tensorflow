# -*- coding utf-8 -*-
# C:\Users\lxg\Documents\Python
# Author:李小根
# Time:2019/1/22
import tensorflow as tf

# 创建节点
w = tf.Variable([.1], dtype=tf.float32)
b = tf.Variable([-.1], dtype=tf.float32)
x = tf.placeholder(tf.float32)  # 输入数据
y = tf.placeholder(tf.float32)  # 输出数据

linear_model = w * x + b  # 创建线性模型

loss = tf.reduce_mean(tf.square(linear_model - y))  # 损失函数

# 下面使用梯度下降算法来优化模型参数
optimizer = tf.train.GradientDescentOptimizer(0.001)  # 学习率为0.001
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()  # 初始化全部变量
with tf.Session() as sess:
    sess.run(init)
    print(sess.run(linear_model, {x: [1, 2, 3, 6, 8]}))
    print('初始值时的损失函数：', sess.run(loss, {x: [1, 2, 3, 6, 8], y: [4.8, 8.5, 10.4, 21.0, 25.3]}))

    for i in range(10000):
        sess.run(train, {x: [1, 2, 3, 6, 8], y: [4.8, 8.5, 10.4, 21.0, 25.3]})
    print('使用梯度下降算法:\nw:%s\tb:%s\tloss:%s' %
          (sess.run(w), sess.run(b), sess.run(loss, {x: [1, 2, 3, 6, 8], y: [4.8, 8.5, 10.4, 21.0, 25.3]})))
