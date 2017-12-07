# -*- coding: UTF-8 -*-
import tensorflow as tf
import slim as slim2
tf.logging.set_verbosity(tf.logging.WARN)


def transit_layer(x,name,depth,pool_stride,activation_fn=tf.nn.relu,is_training=False):
    with tf.variable_scope(name):
        x = slim2.conv2d(x, 'conv', depth,[1,1], 1, activation_fn=None)
        x = slim2.batch_norm('bn', x, is_training)
        x = activation_fn(x)
        k=pool_stride
        pool = tf.nn.max_pool(x, ksize=[1, k+1, k+1, 1], strides=[1, k, k, 1], padding='SAME', name='pool')
        return pool


def block(x, name, depth, activation_fn=tf.nn.relu,is_training=False):
    """ res改进版
    1,block里不使用bn, transit才使用bn,
    2,只有一个shortcut
    out_depth=in_depth+depth,
    size不变，out_size=in_size
    """
    with tf.variable_scope(name):
        orig_x=x
        x = slim2.conv2d(x, 'conv1', depth,[1,1], stride=1, activation_fn=activation_fn)
        x = slim2.conv2d(x, 'conv2', depth,[3,3], stride=1, activation_fn=activation_fn)
        x = slim2.conv2d(x, 'conv3', depth,[3,3], stride=1, activation_fn=activation_fn)
        y= tf.concat([x,orig_x],3)
        return y


def inference(input_x,is_training=False):
    #
    net_func =block
    activation_fn=tf.nn.relu
    with tf.variable_scope("layer1") :
        # 在cifar10数据集上，5x5,strdie=1的效果要好些。
        conv1 = slim2.conv2d(input_x, "conv1", 64, [5,5], 1, activation_fn=tf.nn.relu)
        net = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')

    # res1
    net = net_func(net,"block1_1", 64, activation_fn, is_training)
    net = net_func(net,"block1_2", 64, activation_fn, is_training)
    net = net_func(net,"block1_3", 64, activation_fn, is_training)
    #
    net = transit_layer(net,"transit1",64,2,activation_fn, is_training)
    net = net_func(net,"block2_1", 64, activation_fn, is_training)
    net = net_func(net,"block2_2", 128, activation_fn, is_training)
    net = net_func(net,"block2_3", 128, activation_fn, is_training)
    net = net_func(net,"block2_4", 128, activation_fn, is_training)
    #
    net = transit_layer(net,"transit2",128,2, activation_fn, is_training)
    net = net_func(net,"block3_1", 128, activation_fn, is_training)
    net = net_func(net,"block3_2", 256, activation_fn, is_training)
    net = net_func(net,"block3_3", 256, activation_fn, is_training)
    net = net_func(net,"block3_4", 256, activation_fn, is_training)
    #
    global_avg_pool = slim2.global_avg_pool(net)
    #
    resh1 = slim2.flatten(global_avg_pool)
    #
    if is_training:
        resh1=tf.nn.dropout(resh1, 0.6)
    output=slim2.fully_connected(resh1, "output", 10, None)
    return output


"""
实验结果：
1.训练时运行速度很快
2.relu , acc=0.867
2.swish, acc=0.877
"""




