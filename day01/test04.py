# -*- coding utf-8 -*-
# C:\Users\lxg\Documents\Python
# Author:李小根
# Time:2019/1/21
import tensorflow as tf


# 占位符
a = tf.placeholder(shape=[2, ], dtype=tf.float32, name=None)
b = tf.constant([6, 4], tf.float32)
c = tf.add(a, b)

with tf.Session() as sess:
    filewrite = tf.summary.FileWriter('./tmp/', sess.graph)
    print(sess.run(c, feed_dict={a: [10, 10]}))
