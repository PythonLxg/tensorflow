# -*- coding utf-8 -*-
# C:\Users\lxg\Documents\Python
# Author:李小根
# Time:2019/1/21
import tensorflow as tf

# 变量的创建
a = tf.Variable(1, name='a')
b = tf.Variable([1, 2], name='b')
c = tf.Variable([[0, 1], [2, 3]], name='c')
d = tf.Variable(tf.zeros([10]), name='d')
e = tf.Variable(tf.random_normal([2, 3], mean=-1, stddev=4, seed=1), name='e')

# 初始化全部的变量
init = tf.global_variables_initializer()
# init_sub = tf.variables_initializer([a, b], name='init_sub')  初始化部分的变量
# init_var = tf.Variable(a)  初始化单个变量

# 保存变量
saver = tf.train.Saver()  # 调用Saver()存储器方法

# 变量文件的位置
# model_file = tf.train.latest_checkpoint('./test03/')

# 执行图模型
with tf.Session() as sess:
    sess.run(init)
    # 设置存储路径
    save_path = saver.save(sess, './test03.ckpt')
    # 保存的文件是二进制文件，Saver提供了内置的计数器自动为checkpoint文件编号可以多次保存
    # saver.restore(sess, model_file)  # 恢复变量
