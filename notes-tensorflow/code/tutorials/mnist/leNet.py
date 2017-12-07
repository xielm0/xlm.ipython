# -*- coding:utf-8 -*-
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
SESS_CONFIG=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
SESS_CONFIG.gpu_options.allow_growth = True


def conv_op(input_op, name, ksize, strides):
    with tf.variable_scope(name) as scope:
        n_in = input_op.get_shape()[-1].value
        [kh,kw,n_out]=ksize
        kernel = tf.get_variable("kernel",
                                 shape=[kh, kw, n_in, n_out],
                                 dtype=tf.float32,
                                 initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv = tf.nn.conv2d(input_op, kernel, strides, padding='SAME')
        biases = tf.get_variable('biases', shape=[n_out], initializer=tf.constant_initializer(0),  dtype=tf.float32)
        z = tf.nn.bias_add(conv, biases)
        out = tf.nn.relu(z)
        return out


def fc_op(input_op, name, n_out, activation='relu'):
    with tf.variable_scope(name) as scope:
        n_in = input_op.get_shape()[-1].value

        weights = tf.get_variable("weights",
                                  shape=[n_in, n_out],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases = tf.get_variable('biases', shape=[n_out], initializer=tf.constant_initializer(0),  dtype=tf.float32)
        z=tf.matmul(input_op, weights) + biases
        if activation=='relu':
            out = tf.nn.relu(z)
        elif activation=='softmax':
            out = tf.nn.softmax(z)
        else:
            out=z
        return out


def inference(x, train_flag=False):
    #
    x_image = tf.reshape(x, [-1,28,28,1])
    conv1 = conv_op(x_image,name="conv1", ksize=[5, 5, 32], strides=[1, 1, 1, 1])
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                           padding='SAME', name='pool1')
    conv2 = conv_op(pool1,name="conv2", ksize=[5, 5, 64], strides=[1, 1, 1, 1])
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                           padding='SAME', name='pool2')
    #
    shp = pool2.get_shape()
    flattened_shape = shp[1].value * shp[2].value * shp[3].value
    resh1 = tf.reshape(pool2, [-1, flattened_shape] )
    #
    fc3=fc_op(resh1,name="fc3",n_out=128)
    if train_flag:
        fc3=tf.nn.dropout(fc3, 0.5)

    fc4=fc_op(fc3,name="fc4",n_out=128)
    if train_flag:
        fc4=tf.nn.dropout(fc4, 0.5)

    output=fc_op(fc4,"output", 10, 'softmax')
    return output


def train():
    # build graph
    x = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder(tf.float32, [None, 10])
    y_conv=inference(x)
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
    opt=tf.train.AdamOptimizer(1e-4)
    train_step = opt.minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    #
    mnist = input_data.read_data_sets("/export/Data/MNIST_data", one_hot=True)

    with tf.Session(config=SESS_CONFIG) as sess:
        tf.global_variables_initializer().run()
        for i in range(20000):
            batch = mnist.train.next_batch(50)
            if i%100 == 0:
                _,loss_value,train_accuracy = sess.run([train_step,cross_entropy,accuracy],feed_dict={ x:batch[0], y_: batch[1] })
                print("step %d, loss = %.10f "%(i, loss_value))
                print("step %d, training accuracy %g"%(i, train_accuracy))
            else:
                _ = sess.run(train_step,feed_dict={ x:batch[0], y_: batch[1] })

        test_accuracy = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
        print("test accuracy %g" %test_accuracy)

if __name__ == '__main__':
    train()



