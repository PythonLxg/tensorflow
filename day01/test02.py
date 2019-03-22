# -*- coding utf-8 -*-
# C:\Users\lxg\Documents\Python
# Author:李小根
# Time:2019/1/21
import tensorflow as tf


a = tf.zeros([2, 3], dtype=tf.float32, name='a')  # tf.ones(), tf.fill(), tf.range(), tf.linspace()类似
b = tf.random_normal([2, 3], mean=-1, stddev=4, dtype=tf.float32, seed=1, name='b')
# tf.truncates_normal(), tf.random_uniform(), tf.random_shuffle()类似

with tf.Session() as sess:
    print(sess.run(a))
    print(sess.run(b))
