# -*- coding utf-8 -*-
# C:\Users\lxg\Documents\Python
# Author:李小根
# Time:2019/1/21
import tensorflow as tf


sayHi = tf.constant('Hello World')

m1 = tf.constant([[3, 5]], shape=(1, 2), dtype=tf.int32)
m2 = tf.constant([[2, 4]])
m3 = tf.constant([[1], [4]])
add = tf.add(m1, m2)
mul = tf.matmul(m2, m3, name='mul')

with tf.Session() as sess:
    res01 = sess.run(sayHi)
    res02 = sess.run(add)
    res03 = sess.run(mul)

    print(res01)
    print(add)
    print(res02)
    print(add.eval())
    print(mul)
    print(res03)
