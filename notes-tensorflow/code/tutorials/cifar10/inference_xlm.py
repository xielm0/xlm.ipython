# -*- coding: UTF-8 -*-
import tensorflow as tf
import slim as slim2
tf.logging.set_verbosity(tf.logging.WARN)


def transit(x,name,out_depth,stride):
    with tf.variable_scope(name):
        x = slim2.conv2d(x,"conv1",out_depth,[1,1],stride=1)
        y = tf.nn.max_pool(x, ksize=[1, stride, stride, 1], strides=[1, stride, stride, 1], padding='SAME', name='pool1')
        return y


def block(x, name, mid_depth, is_training=False):
    """residual unit with 3 sub layers.
    与普通的res不同的是，这里的out_depth=in_depth*2, out_size=in_size

    """
    activation_fn=tf.nn.relu
    with tf.variable_scope(name):
        # in_depth = x.get_shape()[-1].value
        orig_x=x
        x = slim2.conv2d(x, 'conv1', mid_depth,[1,1], stride=1, activation_fn=activation_fn)
        x = slim2.conv2d(x, 'conv2', mid_depth,[3,3], stride=1, activation_fn=activation_fn)
        x = slim2.conv2d(x, 'conv3', mid_depth,[3,3], stride=1, activation_fn=activation_fn)
        # x = slim2.batch_norm('bn1', x, is_training)
        y= tf.concat([x,orig_x],3)
        return y


def inference(input_x,is_training=False):
    #
    net_func =block
    with tf.variable_scope("layer1") :
        # 在cifar10数据集上，5x5,strdie=1的效果要好些。
        conv1 = slim2.conv2d(input_x, "conv1", 64, [5,5], 1, activation_fn=tf.nn.relu)
        net = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')

    # res1
    net = net_func(net,"block1_1", 64, is_training)
    net = net_func(net,"block1_2", 64, is_training)
    net = net_func(net,"block1_3", 64, is_training)
    net = tf.nn.max_pool(net, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')
    #
    net = net_func(net,"block2_1", 256, is_training)
    net = net_func(net,"block2_2", 256, is_training)
    net = net_func(net,"block2_3", 256, is_training)
    net = net_func(net,"block2_4", 256, is_training)
    # net = tf.nn.max_pool(net, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')

    #
    global_avg_pool = slim2.global_avg_pool(net)
    #
    resh1 = slim2.flatten(global_avg_pool)
    #
    if is_training:
        resh1=tf.nn.dropout(resh1, 0.6)
    output=slim2.fully_connected(resh1, "output", 10, None)
    return output







