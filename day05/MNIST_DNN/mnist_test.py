# -*- coding utf-8 -*-
# C:\Users\lxg\Documents\Python
# Author:李小根
# Time:2019/2/23
import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import mnist_forward
import mnist_train

# 每10秒加载一次最新的模型
EVAL_INTERVAL_SECS = 10


def evaluate(mnist):
    with tf.Graph().as_default() as _:
        x = tf.placeholder(tf.float32, [None, mnist_forward.INPUT_NODE],
                           name='x-input')
        y_ = tf.placeholder(tf.float32, [None, mnist_forward.OUTPUT_NODE],
                            name='y-input')
        validate_feed = {x: mnist.validation.images,
                         y_: mnist.validation.labels}

        y = mnist_forward.forward(x, None)

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # 通过变量重命名来加载模型
        # 前向传播过程
        variable_averages = tf.train.ExponentialMovingAverage(mnist_train.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        while True:
            with tf.Session() as sess:
                # 找到目录中的最新模型文件名
                ckpt = tf.train.get_checkpoint_state(mnist_train.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    # 加载模型
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    # 通过文件名得到模型保存时迭代的轮数
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    accuracy_score = sess.run(accuracy, feed_dict=validate_feed)
                    print("After %s training step, accuracy = %g" % (global_step, accuracy_score))
                else:
                    print("No checkpoint file found")
                    return
            time.sleep(EVAL_INTERVAL_SECS)


def main(argv=None):
    mnist = input_data.read_data_sets('/data', one_hot=True)
    evaluate(mnist)


if __name__ == '__main__':
    tf.app.run()
