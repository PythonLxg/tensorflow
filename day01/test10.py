# -*- coding utf-8 -*-
# C:\Users\lxg\Documents\Python
# Author:李小根
# Time:2019/2/20
# import tensorflow as tf
# import os
# import time
#
# # 生成 TFRecords 文件
# def main(unused_argv):
#     # 获取数据
#     data_sets = mnist.read_data_sets(FLAGS.directory,
#                                      dtype=tf.uint8,  # 注意，这里的编码是 uint8
#                                      reshape=False,
#                                      validation_size=FLAGS.validation_size)
#     # 将数据转换为 tf.train.Example 类型，并写入 TFRecords 文件
#     convert_to(data_sets.train, 'train')
#     convert_to(data_sets.validation, 'validation')
#     convert_to(data_sets.test, 'test')
#
#
# """
# 转换函数 convert_to 的主要功能是，将数据填入到 tf.train.Example 的协议缓冲区（protocol
# buffer）中，将协议缓冲区序列化为一个字符串，通过 tf.python_io.TFRecordWriter 写入 TFRecords文件。
# """
#
#
# def convert_to(data_set, name):
#     images = data_set.images
#     labels = data_set.labels
#     num_examples = data_set.num_examples  # 55000个训练数据，5000个验证数据，10000个测试数据
#     if images.shape[0] != num_examples:
#         raise ValueError('Images size %d does not match label size %d.' %
#                          (images.shape[0], num_examples))
#
#     rows = images.shape[1]  # 28
#     cols = images.shape[2]  # 28
#     depth = images.shape[3]  # 1，是黑白图像，所以是单通道
#
#     filename = os.path.join(FLAGS.directory, name + '.tfrecords')
#     print('Writing', filename)
#     writer = tf.python_io.TFRecordWriter(filename)
#     for index in range(num_examples):
#         image_raw = images[index].tostring()
#     # 写入协议缓冲区中，height、width、depth、label 编码成int64 类型，image_raw 编码成二进制
#     example = tf.train.Example(features=tf.train.Features(feature={
#         'height': _int64_feature(rows),
#         'width': _int64_feature(cols),
#         'depth': _int64_feature(depth),
#         'label': _int64_feature(int(labels[index])),
#         'image_raw': _bytes_feature(image_raw)}))
#     writer.write(example.SerializeToString())  # 序列化为字符串
#     writer.close()
#
#
# def _int64_feature(value):
#     return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
#
#
# def _bytes_feature(value):
#     return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
#
#
# # 运行结束后，在/tmp/data 下生成3 个文件，即train.tfrecords、validation.tfrecords 和test.tfrecords
#
#
# # 从队列中读取
# """
# 一旦生成了 TFRecords 文件，接下来就可以使用队列读取数据了。主要分为 3 步：
# （1）创建张量，从二进制文件读取一个样本；
# （2）创建张量，从二进制文件随机读取一个 mini-batch；
# （3）把每一批张量传入网络作为输入节点
# """
#
#
# def read_and_decode(filename_queue):  # 输入文件名队列
#     reader = tf.TFRecordReader()
#     _, serialized_example = reader.read(filename_queue)
#
#     features = tf.parse_single_example(  # 解析 example
#         serialized_example,
#         # 必须写明 features 里面的 key 的名称
#         features={
#             'image_raw': tf.FixedLenFeature([], tf.string),  # 图片是 string 类型
#             'label': tf.FixedLenFeature([], tf.int64),  # 标记是 int64 类型
#         })
#
#     # 对于 BytesList，要重新进行解码，把 string 类型的 0 维 Tensor 变成 uint8 类型的一维 Tensor
#     image = tf.decode_raw(features['image_raw'], tf.uint8)
#     image.set_shape([mnist.IMAGE_PIXELS])
#     # Tensor("input/DecodeRaw:0", shape=(784,), dtype=uint8)
#     # image 张量的形状为：Tensor("input/sub:0", shape=(784,), dtype=float32)
#     image = tf.cast(image, tf.float32) * (1． / 255) - 0.5
#     # 把标记从 uint8 类型转换为 int32 类型
#     # label 张量的形状为 Tensor("input/Cast_1:0", shape=(), dtype=int32)
#     label = tf.cast(features['label'], tf.int32)
#     return image, label
#
#
# # 接下来使用tf.train.shuffle_batch将前面生成的样本随机化，获得一个最小批次的张量：
#
# def inputs(train, batch_size, num_epochs):
#     # 输入参数:
#     # train: 选择输入训练数据/验证数据
#     # batch_size: 训练的每一批有多少个样本
#     # num_epochs: 过几遍数据，设置为 0/None 表示永远训练下去
#     """
#     返回结果：A tuple (images, labels)
#     * images: 类型 float, 形状[batch_size, mnist.IMAGE_PIXELS]，范围[-0.5, 0.5].
#     * labels： 类型 int32，形状[batch_size]，范围 [0, mnist.NUM_CLASSES]
#     注意 tf.train.QueueRunner 必须用 tf.train.start_queue_runners()来启动线程
#     """
#     if not num_epochs: num_epochs = None
#     # 获取文件路径，即/tmp/data/train.tfrecords, /tmp/data/validation.records
#     filename = os.path.join(FLAGS.train_dir,
#                             TRAIN_FILE if train else VALIDATION_FILE)
#     with tf.name_scope('input'):
#     # tf.train.string_input_producer 返回一个 QueueRunner，里面有一个 FIFOQueue
#     filename_queue = tf.train.string_input_producer(
#         [filename], num_epochs=num_epochs)  # 如果样本量很大，可以分成若干文件，把文件名列表传入
#     image, label = read_and_decode(filename_queue)
#     # 随机化 example，并把它们规整成 batch_size 大小
#     # tf.train.shuffle_batch 生成了 RandomShuffleQueue，并开启两个线程
#     images, sparse_labels = tf.train.shuffle_batch(
#         [image, label], batch_size=batch_size, num_threads=2,
#         capacity=1000 + 3 * batch_size,
#         min_after_dequeue=1000)  # 留下一部分队列，来保证每次有足够的数据做随机打乱
#
#
#     return images, sparse_labels
#
#
# def run_training():
#     with tf.Graph().as_default():
#         # 输入 images 和 labels
#         images, labels = inputs(train=True, batch_size=FLAGS.batch_size,
#                                 num_epochs=FLAGS.num_epochs)
#         # 构建一个从推理模型来预测数据的图
#         logits = mnist.inference(images,
#                                  FLAGS.hidden1,
#                                  FLAGS.hidden2)
#         loss = mnist.loss(logits, labels)  # 定义损失函数
#         # Add to the Graph operations that train the ckpt.
#         train_op = mnist.training(loss, FLAGS.learning_rate)
#         # 初始化参数，特别注意：string_input_producer 内部创建了一个 epoch 计数变量，
#         # 归入tf.GraphKeys.LOCAL_VARIABLES集合中，必须单独用initialize_local_variables()初始化
#         init_op = tf.group(tf.global_variables_initializer(),
#                            tf.local_variables_initializer())
#
#         sess = tf.Session()
#
#         sess.run(init_op)
#
#         # Start input enqueue threads.
#         coord = tf.train.Coordinator()
#         threads = tf.train.start_queue_runners(sess=sess, coord=coord)
#
#         try:
#             step = 0
#             while not coord.should_stop():  # 进入永久循环
#                 start_time = time.time()
#                 _, loss_value = sess.run([train_op, loss])
#             duration = time.time() - start_time
#
#         # 每 100 次训练输出一次结果
#             if step % 100 == 0:
#                 print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
#             step += 1
#         except tf.errors.OutOfRangeError:
#             print('Done training for %d epochs, %d steps.' % (FLAGS.num_epochs, step))
#         finally:
#             coord.request_stop()  # 通知其他线程关闭
#
#         coord.join(threads)
#         sess.close()
