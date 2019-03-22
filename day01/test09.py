# -*- coding utf-8 -*-
# C:\Users\lxg\Documents\Python
# Author:李小根
# Time:2019/2/20
import tensorflow as tf
import os

# 加载数据
mnist = tf.input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
X = tf.placeholder("float", [None, 784])
Y = tf.placeholder("float", [None, 10])
# 初始化权重参数
w_h = tf.init_weights([784, 625])
w_h2 = tf.init_weights([625, 625])
w_o = tf.init_weights([625, 10])


# 定义权重函数
def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


# 定义模型
def model(X, w_h, w_h2, w_o, p_keep_input, p_keep_hidden):
    # 第一个全连接层
    X = tf.nn.dropout(X, p_keep_input)
    h = tf.nn.relu(tf.matmul(X, w_h))
    h = tf.nn.dropout(h, p_keep_hidden)
    # 第二个全连接层
    h2 = tf.nn.relu(tf.matmul(h, w_h2))
    h2 = tf.nn.dropout(h2, p_keep_hidden)
    return tf.matmul(h2, w_o)  # 输出预测值


# 生成网络模型，得到预测值，代码如下:
p_keep_input = tf.placeholder("float")
p_keep_hidden = tf.placeholder("float")
py_x = model(X, w_h, w_h2, w_o, p_keep_input, p_keep_hidden)
# 定义损失函数，代码如下：
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x, Y))
train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
predict_op = tf.argmax(py_x, 1)

# 首先，我们定义一个存储路径，这里就用当前路径下的 ckpt_dir 目录，代码如下：
ckpt_dir = "./ckpt_dir"
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)
# 定义一个计数器，为训练轮数计数，代码如下：
# 计数器变量，设置它的 trainable=False，不需要被训练
global_step = tf.Variable(0, name='global_step', trainable=False)

# 在声明完所有变量后，调用 tf.train.Saver
saver = tf.train.Saver()
# 位于 tf.train.Saver 之后的变量将不会被存储
non_storable_variable = tf.Variable(777)
# 训练模型并存储，如下：
with tf.Session() as sess:
    tf.initialize_all_variables().run()
    start = global_step.eval()  # 得到 global_step 的初始值
    print("Start from:", start)
    for i in range(start, 100):
        # 以 128 作为 batch_size
        for start, end in zip(range(0, len(trX), 128), range(128, len(trX) + 1, 128)):
            sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end],
                                          p_keep_input: 0.8, p_keep_hidden: 0.5})
        global_step.assign(i).eval()  # 更新计数器
        saver.save(sess, ckpt_dir + "/ckpt.ckpt", global_step=global_step)  # 存储模型

# 如果有已经训练好的模型变量文件，可以用 saver.restore 来进行模型加载
# with tf.Session() as sess:
# tf.initialize_all_variables().run()
# ckpt = tf.train.get_checkpoint_state(ckpt_dir)
# if ckpt and ckpt.model_checkpoint_path:
# print(ckpt.model_checkpoint_path)
# saver.restore(sess, ckpt.model_checkpoint_path) # 加载所有的参数
# 从这里开始就可以直接使用模型进行预测，或者接着继续训练了
