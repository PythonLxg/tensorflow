# -*- coding utf-8 -*-
# C:\Users\lxg\Documents\Python
# Author:李小根
# Time:2019/1/22
import numpy as np
import tensorflow as tf


# 自定义Estimator
# 定义模型训练函数，同时也定义了特征向量
def model_fn(features, labels, mode):
    # 构建线性模型
    w = tf.get_variable('W', [1], dtype=tf.float64)
    b = tf.get_variable('b', [1], dtype=tf.float64)
    y = w * features['x'] + b
    # 构建损失模型
    loss = tf.reduce_sum(tf.square(y - labels))
    # 训练模型子图
    global_step = tf.train.get_global_step()
    optimizer = tf.train.GradientDescentOptimizer(0.001)
    train = tf.group(optimizer.minimize(loss), tf.assign_add(global_step, 1))
    # 通过EstimatorSpec指定我们的训练子图积极损失模型
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=y,
        loss=loss,
        train_op=train
    )


# 创建自定义的训练模型
estimator = tf.estimator.Estimator(model_fn=model_fn)

x_train = np.array([1., 2., 3., 6., 8.])
y_train = np.array([4.8, 8.5, 10.4, 21.0, 25.3])

x_eval = np.array([2., 5., 7., 9.])
y_eval = np.array([7.6, 17.2, 23.6, 28.8])

train_input_fn = tf.estimator.inputs.numpy_input_fn(
    {'x': x_train}, y_train, batch_size=2, num_epochs=None, shuffle=True)
train_input_fn_2 = tf.estimator.inputs.numpy_input_fn(
    {'x': x_train}, y_train, batch_size=2, num_epochs=1000, shuffle=False)
eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    {'x': x_eval}, y_eval, batch_size=2, num_epochs=1000, shuffle=False)

estimator.train(input_fn=train_input_fn, steps=1000)

train_metrics = estimator.evaluate(input_fn=train_input_fn_2)
print('train metrics:', train_metrics)
eval_metrics = estimator.evaluate(input_fn=eval_input_fn)
print('eval metrics:', eval_metrics)
