# import  tensorflow as tf
# import  numpy as np
# import  matplotlib.pyplot as plt
#
# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("MNIST_data",one_hot=True )  # 得到数据的地址
#
# IN_SIZE = 784
# LR = 0.5
# BATCH_SIZE = 100
#
# def add_layer(inputs, in_size, out_size, activation_function=None):
#     Weights = tf.Variable(tf.random_normal([in_size, out_size]))  # 默认float32
#     biases = tf.Variable(tf.zeros([1,out_size], tf.float32))  # xw + b
#     Wx_plus_b = tf.add(tf.matmul(inputs, Weights), biases)
#     if activation_function is None:
#         outputs =  Wx_plus_b
#     else:
#         outputs = activation_function(Wx_plus_b)
#     return outputs
#
# def compute_accuarcy(v_xs, v_ys):
#     global prediction
#     y_pre = sess.run(prediction, feed_dict={xs:v_xs})
#     correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1)) # 返回最大值得索引
#     # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#     accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#     result = sess.run(accuracy, feed_dict={xs:v_xs, ys:v_ys})
#     return result
#     # pass
#
# xs = tf.placeholder(tf.float32, [None, IN_SIZE])
# ys = tf.placeholder(tf.float32, [None, 10])
#
# # net = add_layer(xs, IN_SIZE, 1024, activation_function=tf.nn.softmax)
# prediction = add_layer(xs, IN_SIZE, 10, activation_function=tf.nn.softmax)
#
# # ========= loss ============
# # ys * tf.log(prediction)
# # loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),reduction_indices=[1]))
# loss = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
#                                                   reduction_indices=[1]))  # loss
# train_step = tf.train.GradientDescentOptimizer(LR).minimize(loss)
# # ===========================
#
# # batch_xs, batch_ys = mnist.train.next_batch(BATCH_SIZE)
# def show_img(batch_xs,batch_ys):
#     for x,y, i in zip(batch_xs,batch_ys, range(1)):
#         x = x.reshape(28, 28)
#         plt.title(y)
#         plt.imshow(x)
#         plt.show()
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#
#     for i in range(1000):
#         batch_xs, batch_ys = mnist.train.next_batch(BATCH_SIZE)
#         sess.run(train_step, feed_dict={xs:batch_xs, ys:batch_ys})
#         if i % 100 == 0:
#             # print(sess.run(loss,feed_dict={xs:batch_test_xs, ys:batch_test_ys}))
#             batch_test_xs, batch_test_ys = mnist.test.images,  mnist.test.labels
#             print(compute_accuarcy(batch_test_xs, batch_test_ys))
#             # show_img(batch_test_xs, batch_test_ys)
#         #     print(print_acc)
#
# # # 快捷键:
# # F3   下一个,Shift + F3   前一个
# # ctrl + p 看参数定义  , ctrl + q 看文档,


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import os

IN_SIZE = 784
LR = 0.5
BATCH_SIZE = 100

model_path = r'.\weight_file'
graph_path = './log'

if not os.path.exists(model_path):
    os.makedirs(model_path)
#
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
test_x = mnist.test.images[:3000]
test_y = mnist.test.labels[:3000]


def add_layer(inputs, in_size, out_size, activitation_funcation=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.add(tf.matmul(inputs, Weights), biases)

    if activitation_funcation is None:
        outputs = Wx_plus_b
    else:
        outputs = activitation_funcation(Wx_plus_b)
    return outputs


xs = tf.placeholder(tf.float32, [None, IN_SIZE])
ys = tf.placeholder(tf.float32, [None, 10])

prediction = add_layer(xs, IN_SIZE, 10, tf.nn.softmax)


# def cnn_layer(inputs, ):
#     net = tf.layers.conv2d(
#         inputs=inputs,
#         filters=32,
#         kernel_size=[5, 5],
#         strides=1,
#         padding='same',  # 进行填充, same 就是输出size 和输入一样, value 是会缩小
#         activation=tf.nn.relu,
#     )
#
#     net = tf.layers.max_pooling2d(net, [2, 2], 2)  # 14x14x32
#     net = tf.layers.conv2d(net, 64, [5, 5], 1, 'same', tf.nn.relu)
#     net = tf.layers.max_pooling2d(net, [2, 2], 2, )  # 7x7x64
#     flat = tf.reshape(net, [-1, 7 * 7 * 64])  # 扁平化
#     net = tf.layers.dense(flat, 1024, tf.nn.relu)
#     dropout = tf.layers.dropout(net, rate=0.5, )
#     logits = tf.layers.dense(dropout, 10, )
#
#     loss = tf.losses.softmax_cross_entropy(onehot_labels=, logits=logits)
#     train_step = tf.train.AdamOptimizer(LR).minimize(loss)
#     accuracy = tf.metrics.accuracy(labels=, predictions=tf.argmax(logits, 1))[0]


def img_show(xs, ys):
    for x, y in zip(xs, ys):
        x = x.reshape(28, 28)
        plt.title(y)
        plt.imshow(x)
        plt.show()


def train():
    # 损失函数的选择
    # 方法1
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(
        ys * tf.log(prediction), reduction_indices=[1]))
    train_step = tf.train.GradientDescentOptimizer(LR).minimize(cross_entropy)

    # 方法2
    # cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = prediction, labels = ys))
    # train_step = tf.train.AdamOptimizer(LR).minimize(cross_entropy)

    # 方法3
    # loss = tf.reduce_mean(tf.square(ys - prediction), reduction_indices=[1])

    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(prediction, 1), tf.argmax(ys, 1)), tf.float32))

    # 数据保存器
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter(graph_path, sess.graph)
        for i in range(1000):
            batch_xs, batch_ys = mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys})
            if i % 50 == 0:
                batch_test_xs, batch_test_ys = mnist.test.images, mnist.test.labels
                # img_show(batch_test_xs, batch_test_ys)
                loss = sess.run(cross_entropy, feed_dict={xs: batch_test_xs, ys: batch_test_ys})
                acc = accuracy.eval({xs: batch_test_xs, ys: batch_test_ys})  # 只使用一个测试数据进行测试
                print("acc:", acc, "  loss:", loss)

                saver.save(sess, model_path + '/train.model')
        print('(〃＞皿＜)  训练完了')


def test(x_test, y_test):
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1]))
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(prediction, 1), tf.argmax(ys, 1)), tf.float32))
    # return loss, accuracy

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, model_path + '/train.model')

        for i in range(10):
            acc, loss = sess.run([accuracy, cross_entropy], feed_dict={xs: x_test, ys: y_test})
            # img_show( mnist.test.images,  mnist.test.labels)
            print("acc: ", acc, "  loss:", loss)
        print('(ｷ｀ﾟДﾟ´)!!   测试完了')


def mnist_recognition(x_test):
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, model_path + '/train.model')
        pre_number = sess.run(prediction, feed_dict={xs: x_test})  # return numpy.ndarray
        pre_list = pre_number.tolist()
        pre_result = str(pre_list.index(max(pre_list)))
        # pre_result =
        img_show(x_test, pre_result)


if __name__ == '__main__':
    train()
    # test()
    # mnist_recognition(mnist.test.images)
