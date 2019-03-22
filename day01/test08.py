# -*- coding utf-8 -*-
# C:\Users\lxg\Documents\Python
# Author:李小根
# Time:2019/1/22
import tensorflow as tf

# 创建节点
W = tf.Variable([.1], dtype=tf.float32, name='W')
b = tf.Variable([-.1], dtype=tf.float32, name='b')
x = tf.placeholder(tf.float32, name='x')  # 输入数据
y = tf.placeholder(tf.float32, name='y')  # 输出数据

linear_model = W * x + b  # 创建线性模型

# 将损失模型隐藏到loss-model模块
with tf.name_scope('loss_model'):
    loss = tf.reduce_sum(tf.square(linear_model - y))  # 损失函数
    # 给损失模型的输出添加scalar，用来观察loss的收敛曲线
    tf.summary.scalar('loss', loss)

# 下面使用梯度下降算法来优化模型参数
optimizer = tf.train.GradientDescentOptimizer(0.001)  # 学习率为0.001
train = optimizer.minimize(loss)

x_train = [1, 2, 3, 6, 8]
y_train = [4.8, 8.5, 10.4, 21.0, 25.3]

init = tf.global_variables_initializer()  # 初始化全部变量
with tf.Session() as sess:
    sess.run(init)

    merged = tf.summary.merge_all()  # 调用merge_all()收集所有的操作数据

    # 模型运行产生的所有数据保存到 ./test08_tmp文件夹供TensorBoard使用
    writer = tf.summary.FileWriter(r'test08_tmp', sess.graph)

    for i in range(10000):
        # 训练时传入merge
        summary, _ = sess.run([merged, train], {x: x_train, y: y_train})
        # 收集每次训练产生的数据
        writer.add_summary(summary, i)

    curr_W, curr_b, curr_loss = sess.run(
        [W, b, loss], {x: x_train, y: y_train})

    print("After train W: %s b %s loss: %s" % (curr_W, curr_b, curr_loss))
# 查看tensorboard: tensorboard --logdir test08_tmp
