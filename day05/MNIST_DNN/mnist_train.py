# -*- coding utf-8 -*-
# C:\Users\lxg\Documents\Python
# Author:李小根
# Time:2019/2/23
import os

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 加载mnist_forward.py中定义的常量和前向传播的函数
import mnist_forward

# 配置神经网络的参数
BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARAZTION_RATE = 0.0001
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99
# 模型保存的路径和文件名
MODEL_SAVE_PATH = "./ckpt"
MODEL_NAME = "ckpt.ckpt"


def train(mnist):
    # 定义输入和输出
    x = tf.placeholder(tf.float32, [None, mnist_forward.INPUT_NODE], name="x-input")
    y_ = tf.placeholder(tf.float32, [None, mnist_forward.OUTPUT_NODE], name="y-input")  # 真实值

    regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)
    y = mnist_forward.forward(x, regularizer)  # 预测值

    global_step = tf.Variable(0, trainable=False)

    # 定义损失函数，学习率，滑动平均和训练过程
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variable_averages_op = variable_averages.apply(tf.trainable_variables())
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE, global_step,
        mnist.train.num_examples / BATCH_SIZE,
        LEARNING_RATE_DECAY
    )
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    with tf.control_dependencies([train_step, variable_averages_op]):
        train_op = tf.no_op(name='train')

    # 初始化TensorFlow持久化类
    saver = tf.train.Saver()
    init_op = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init_op)

        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            _, loss_value, step = sess.run([train_op, loss, global_step],
                                           feed_dict={x: xs, y_: ys})
            # 每1000轮保存一次模型
            if i % 1000 == 0:
                print("After %d training steps, batch is %g." % (step, loss_value))
                # 保存当前模型
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME),
                           global_step=global_step)


def main(argv=None):
    mnist = input_data.read_data_sets("/data", one_hot=True)
    train(mnist)


if __name__ == '__main__':
    tf.app.run()
