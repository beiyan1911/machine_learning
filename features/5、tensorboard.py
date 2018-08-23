# TensorBoard 为Tensorflow 的可视化工具，具有很强大的功能。
# TensorBoard 的基本使用步骤是：
# 1、添加summary操作
# 2、用SummaryWriter将这些 summaries 写入一个 log  directory
# 3、用命令启动 TensorBoard ：tensorboard --logdir=/tmp/tensorflow/xxxx

# ## 下面是一个TensorBoard简单示例
# 这个例子用于记录tensorboard的基本用法，来源于 TensorBoard官方示例的演示代码。
# 原代码github地址：https://github.com/dandelionmane/tf-dev-summit-tensorboard-tutorial
#######################################################################################

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('../data/mnist', one_hot=True)
LOGDIR = "../output/board_test/mnist_tutorial/"


# 卷积池化层
def conv_layer(input, size_in, size_out, name="conv"):
    with tf.name_scope(name):
        w = tf.Variable(tf.truncated_normal([5, 5, size_in, size_out], stddev=0.1), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[size_out]), name="B")
        conv = tf.nn.conv2d(input, w, strides=[1, 1, 1, 1], padding="SAME")
        act = tf.nn.relu(conv + b)
        tf.summary.histogram("weights", w)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("activations", act)
        return tf.nn.max_pool(act, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


# 全连接层
def fc_layer(input, size_in, size_out, name="fc"):
    with tf.name_scope(name):
        w = tf.Variable(tf.truncated_normal([size_in, size_out], stddev=0.1), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[size_out]), name="B")
        act = tf.matmul(input, w) + b
        tf.summary.histogram("weights", w)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("activations", act)
        return act


def mnist_model(learning_rate):
    tf.reset_default_graph()
    sess = tf.Session()

    x = tf.placeholder(tf.float32, shape=[None, 784], name="x")
    y = tf.placeholder(tf.float32, shape=[None, 10], name="labels")

    x_image = tf.reshape(x, [-1, 28, 28, 1])

    tf.summary.image('input', x_image, 3)

    # 原始图像： n x 28 x 28 x 1
    conv1 = conv_layer(x_image, 1, 32, "conv1")  # 输出图像尺寸：n x 14 x 14 x 32
    conv_out = conv_layer(conv1, 32, 64, "conv2")  # 输出图像尺寸：n x 7 x 7 x 64

    # 压平
    flattened = tf.reshape(conv_out, [-1, 7 * 7 * 64])

    fc1 = fc_layer(flattened, 7 * 7 * 64, 1024, "fc1")
    relu = tf.nn.relu(fc1)
    tf.summary.histogram("fc1/relu", relu)
    logits = fc_layer(relu, 1024, 10, "fc2")

    # 交叉熵损失
    with tf.name_scope("loss"):
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y), name="loss")
        tf.summary.scalar("loss", loss)

    # 优化
    with tf.name_scope("train"):
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    # 正确率
    with tf.name_scope("accuracy"):
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar("accuracy", accuracy)

    summ = tf.summary.merge_all()

    # 初始化所有参数
    sess.run(tf.global_variables_initializer())
    #
    writer = tf.summary.FileWriter(LOGDIR + "lr_%.0E" % learning_rate)
    writer.add_graph(sess.graph)

    for i in range(2001):
        batch = mnist.train.next_batch(100)
        if i % 5 == 0:
            [train_accuracy, s] = sess.run([accuracy, summ], feed_dict={x: batch[0], y: batch[1]})
            writer.add_summary(s, i)
        sess.run(train_step, feed_dict={x: batch[0], y: batch[1]})


def main():
    # 可以添加其他学习率
    for learning_rate in [1E-3, 1E-4]:
        print('Starting run for Learning Rate：%s' % learning_rate)
        mnist_model(learning_rate)
    print('执行完成! Run `tensorboard --logdir=%s --host=localhost` to see the results.' % LOGDIR)


if __name__ == '__main__':
    main()
