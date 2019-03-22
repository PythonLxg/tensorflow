# -*- coding utf-8 -*-
# C:\Users\lxg\Documents\Python
# Author:李小根
# Time:2019/1/22
import numpy as np
import tensorflow as tf


# 创建特征向量列表
feature_columns = [tf.feature_column.numeric_column('x', shape=[1])]  # 名称为x，只有一个元素的特征向量

# 创建一个LinearRegressor训练器，并传入特征向量列表
estimator = tf.estimator.LinearRegressor(feature_columns=feature_columns)

# 保存训练用的数据
x_train = np.array([1., 2., 3., 6., 8.])
y_train = np.array([4.8, 8.5, 10.4, 21.0, 25.3])

# 保存评估用的数据
x_eval = np.array([2., 5., 7., 9.])
y_eval = np.array([7.6, 17.2, 23.6, 28.8])

# 用训练数据创建一个输入模型，用来进行后面的模型评估
train_input_fn = tf.estimator.inputs.numpy_input_fn({'x': x_train}, y_train,
                                                    batch_size=2, num_epochs=None, shuffle=True)
# 再用训练数据创建一个输入模型，用来进行后面的模型评估
train_input_fn_2 = tf.estimator.inputs.numpy_input_fn({'x': x_train}, y_train,
                                                      batch_size=2, num_epochs=1000, shuffle=False)
# 用评估数据创建一个输入模型，用来进行后面的模型评估
eval_input_fn = tf.estimator.inputs.numpy_input_fn({'x': x_eval}, y_eval,
                                                   batch_size=2, num_epochs=1000, shuffle=False)

# 使用训练数据训练1000次
estimator.train(input_fn=train_input_fn, steps=1000)

# 使用原来训练数据评估一下模型，目的是查看训练的结果
train_metrics = estimator.evaluate(input_fn=train_input_fn_2)
print('train metrics:', train_metrics)

# 使用评估数据评估一下模型，目的是验证模型的泛化性能
eval_metrics = estimator.evaluate(input_fn=eval_input_fn)
print('eval metrics:', eval_metrics)
