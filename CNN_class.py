# import tensorflow as tf
# import numpy as np
# from tensorflow.examples.tutorials.mnist import input_data
#
# mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
# test_x = mnist.test.images[:2000]
# test_y = mnist.test.labels[:2000]
#
# model_path = r'.\weight_file'
#
# LR = 0.01
# STEP = 600
#
# xs = tf.placeholder(tf.float32, [None, 784])
# ys = tf.placeholder(tf.float32, [None, 10])
#
# def train(x_train, y_train):
#     xs_4d = tf.reshape(x_train, [-1, 28, 28, 1])     # 需要2d转4d
#     net = tf.layers.conv2d(xs_4d, 32, 5, strides=1, padding='same', activation=tf.nn.relu)  # 28x28x32
#     net = tf.layers.max_pooling2d(net, 2, 2, )                 # 14x14x32
#     net = tf.layers.conv2d(net, 64, 5, strides=1, padding='same', activation=tf.nn.relu) # 14x14x64
#     net = tf.layers.max_pooling2d(net, 2, 2)                  # 7x7x64
#     flat = tf.reshape(net, [-1, 7 * 7 * 64])
#     net = tf.layers.dense(flat,  1024, activation=tf.nn.relu )
#     drop = tf.layers.dropout(net, 0.5, )
#     logits = tf.layers.dense(drop, 10, )
#
#     loss = tf.losses.softmax_cross_entropy(onehot_labels=y_train, logits=logits, )
#     train_step = tf.train.AdamOptimizer(LR).minimize(loss)
#
#     accuracy = tf.metrics.accuracy(labels=tf.argmax(y_train, axis=1),  # y_train 为 (100, 10)  y_train[0]是样本个数, axis=1 是预测值
#                               predictions=tf.argmax(logits, axis=1))[1]  # accuracy 有两个返回值
#
#     saver = tf.train.Saver()
#     with tf.Session() as sess:
#         sess.run(tf.global_variables_initializer())
#         sess.run(tf.local_variables_initializer())  # 使用 本地初始化  必须要!!!
#         for i in range(STEP):
#             batch_xs, batch_ys = mnist.train.next_batch(100)
#             _, loss_ = sess.run([train_step, loss], feed_dict={xs: batch_xs, ys: batch_ys})
#             if i % 50 == 0:
#                 acc = sess.run(accuracy, feed_dict={xs: test_x, ys: test_y})
#                 print('Step:', i, '| train loss: %.4f' % loss_, '| test accuracy: %.2f' % acc)
#
#                 saver.save(sess, model_path + '/cnn.module')
#         print('(〃＞皿＜)  训练完了')
#
# if __name__ == '__main__':
#     train(xs, ys)
#

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

test_x = mnist.test.images[:2000]
test_y = mnist.test.labels[:2000]

STEP = 1000
BATCH_SIZE = 100
IMAGE_SIZE = 784
LR = 0.01

xs = tf.placeholder(tf.float32, [None, IMAGE_SIZE])
ys = tf.placeholder(tf.float32, [None, 10])

def train(train_x, train_y):
    x_4d = tf.reshape(train_x, [-1, 28, 28, 1] )
    net = tf.layers.conv2d(x_4d, 32, 5, 1, padding='same', activation=tf.nn.relu)
    net = tf.layers.max_pooling2d(net, 2, 2,)
    net = tf.layers.conv2d(net, 64, 5, 1, padding='same', activation=tf.nn.relu )
    net = tf.layers.max_pooling2d(net, 2, 2, )
    flat = tf.reshape(net, [-1, 7 * 7 * 64])
    net = tf.layers.dense(flat, 1024, activation=tf.nn.relu)
    logits = tf.layers.dense(net, 10,)

    loss = tf.losses.softmax_cross_entropy(onehot_labels=train_y, logits=logits, )
    train_setp = tf.train.AdamOptimizer(LR).minimize(loss)

    accuracy = tf.metrics.accuracy(labels=tf.argmax(logits, axis=1), predictions=tf.argmax(train_y, axis=1))[1]

    with tf.Session() as sess:
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())

        for i in range(STEP):
            batch_x, batch_y = mnist.train.next_batch(BATCH_SIZE)  # 设置100个样本
            loss_, _ = sess.run([loss, train_setp], feed_dict={xs: batch_x, ys: batch_y})

            if i % 50 == 0:
                acc = sess.run(accuracy, feed_dict={xs: test_x, ys: test_y})
                print('Step:', i, '| train loss: %.4f' % loss_, '| test accuracy: %.2f' % acc)

if __name__ == '__main__':
    train(xs, ys)

