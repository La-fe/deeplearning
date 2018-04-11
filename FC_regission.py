import  tensorflow as tf
import  numpy as np
import  matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import  os

IN_SIZE = 784
LR = 0.01
BATCH_SIZE = 100

model_path = r'.\weight_file'
if not os.path.exists(model_path):
    os.makedirs(model_path)
#
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

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

def img_show(xs, ys):
    for x, y, i in zip(xs, ys, range(1)):
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
        for i in range(1000):
            batch_xs, batch_ys = mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys})
            if i % 50 == 0:
                batch_test_xs, batch_test_ys = mnist.test.images, mnist.test.labels
                # img_show(batch_test_xs, batch_test_ys)
                loss = sess.run(cross_entropy, feed_dict={xs: batch_test_xs, ys: batch_test_ys})
                acc = accuracy.eval({xs: batch_test_xs, ys: batch_test_ys})  # 只使用一个测试数据进行测试
                print( "acc:", acc, "  loss:", loss)

                saver.save(sess, model_path + '/train.model')
        print('(〃＞皿＜)  训练完了')

def test(x_test, y_test):
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1]))
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(prediction, 1), tf.argmax(ys, 1)), tf.float32))

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, model_path + '/train.model')

        for i in range(10):

            acc, loss = sess.run([accuracy, cross_entropy], feed_dict={xs: x_test, ys: y_test})
            img_show( mnist.test.images,  mnist.test.labels)
            print("acc: ", acc , "  loss:", loss)
        print('(ｷ｀ﾟДﾟ´)!!   测试完了')

def mnist_recognition(x_test):
    with tf.Session() as sess:
        pre_number = sess.run(prediction, feed_dict={xs: x_test})
        img_show(x_test, pre_number)

if __name__ == '__main__':
    # train()
    # test()
    # pass
    mnist_recognition(mnist.test.images)
