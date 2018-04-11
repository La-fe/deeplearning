import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
test_x = mnist.test.images[:2000]
test_y = mnist.test.labels[:2000]

model_path = r'.\weight_file'
graph_path = './log'

LR = 0.01
STEP = 600

with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32, [None, 784])
    x_image = tf.reshape(xs, [-1, 28, 28, 1])
    ys = tf.placeholder(tf.float32, [None, 10])
    tf.summary.image('inputs', x_image, 2)


def train(x_train, y_train):
    with tf.name_scope('cnn'):
        xs_4d = tf.reshape(x_train, [-1, 28, 28, 1])  # 需要2d转4d
        net = tf.layers.conv2d(xs_4d, 32, 5, strides=1, padding='same', activation=tf.nn.relu)  # 28x28x32
        tf.summary.image('conv_1', net[None, None, None, 1], 2)

        net = tf.layers.max_pooling2d(net, 2, 2, )  # 14x14x32
        tf.summary.image('conv_2', net[None, None, None, 1], 2)

        net = tf.layers.conv2d(net, 64, 5, strides=1, padding='same', activation=tf.nn.relu)  # 14x14x64
        tf.summary.image('conv_3', net[None, None, None, 1], 2)

        net = tf.layers.max_pooling2d(net, 2, 2)  # 7x7x64
        tf.summary.image('conv_4', net[None, None, None, 1], 2)

        flat = tf.reshape(net, [-1, 7 * 7 * 64])
        net = tf.layers.dense(flat, 1024, activation=tf.nn.relu)
        drop = tf.layers.dropout(net, 0.5, )
        logits = tf.layers.dense(drop, 10, )
        tf.summary.histogram('output', logits)

    with tf.name_scope('loss'):
        loss = tf.losses.softmax_cross_entropy(onehot_labels=y_train, logits=logits, )
        tf.summary.scalar('loss', loss)
        tf.summary.histogram('loss', loss)

    with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(LR).minimize(loss)

    with tf.name_scope('accuracy'):
        accuracy = tf.metrics.accuracy(labels=tf.argmax(y_train, axis=1),
                                    predictions=tf.argmax(logits, axis=1))[1]  # accuracy 有两个返回值
        tf.summary.scalar('accuracy', accuracy)
        tf.summary.histogram('accuracy', accuracy)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter(graph_path, sess.graph)

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())  # 使用 本地初始化  必须要!!!
        for i in range(STEP):
            batch_xs, batch_ys = mnist.train.next_batch(100)
            _, loss_ = sess.run([train_step, loss], feed_dict={xs: batch_xs, ys: batch_ys})
            if i % 50 == 0:
                result = sess.run(merged, feed_dict={xs: batch_xs, ys: batch_ys})
                writer.add_summary(result, i)

                acc = sess.run(accuracy, feed_dict={xs: test_x, ys: test_y})
                print('Step:', i, '| train loss: %.4f' % loss_, '| test accuracy: %.2f' % acc)

                saver.save(sess, model_path + '/cnn.module')
        print('(〃＞皿＜)  训练完了')


if __name__ == '__main__':
    train(xs, ys)
