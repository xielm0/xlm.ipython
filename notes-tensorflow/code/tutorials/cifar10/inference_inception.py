# -*- coding: UTF-8 -*-
import tensorflow as tf
import slim as slim2
slim = tf.contrib.slim


def inception_v1(input_net, scope_name, depth_list):
    with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                        stride=1, padding='SAME'):
        # mixed: 35 x 35 x 256.
        with tf.variable_scope(scope_name):
            with tf.variable_scope('Branch_0'):
                branch_0 = slim.conv2d(input_net, depth_list[0], [1, 1], scope='Conv2d_0a_1x1')
            with tf.variable_scope('Branch_1'):
                branch_2 = slim.conv2d(input_net, depth_list[1], [1, 1], scope='Conv2d_0a_1x1')
                branch_2 = slim.conv2d(branch_2, depth_list[2], [3, 3], scope='Conv2d_0b_3x3')
            with tf.variable_scope('Branch_2'):
                branch_1 = slim.conv2d(input_net, depth_list[3], [1, 1], scope='Conv2d_0a_1x1')
                branch_1 = slim.conv2d(branch_1, depth_list[4], [3, 5], scope='Conv2d_0b_5x5')
            with tf.variable_scope('Branch_3'):
                branch_3 = slim.avg_pool2d(input_net, [3, 3], scope='AvgPool_0a_3x3')
                branch_3 = slim.conv2d(branch_3, depth_list[5], [1, 1], scope='Conv2d_0b_1x1')
            output_net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
            return output_net


def inception_v2(input_net, scope_name, depth_list):
    with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                        stride=1, padding='SAME'):
        # mixed: 35 x 35 x 256.
        with tf.variable_scope(scope_name):
            with tf.variable_scope('Branch_0'):
                branch_0 = slim.conv2d(input_net, depth_list[0], [1, 1], scope='Conv2d_0a_1x1')
            with tf.variable_scope('Branch_1'):
                branch_2 = slim.conv2d(input_net, depth_list[1], [1, 1], scope='Conv2d_0a_1x1')
                branch_2 = slim.conv2d(branch_2, depth_list[2], [3, 3], scope='Conv2d_0b_3x3')
            with tf.variable_scope('Branch_2'):
                branch_1 = slim.conv2d(input_net, depth_list[3], [1, 1], scope='Conv2d_0a_1x1')
                branch_1 = slim.conv2d(branch_1, depth_list[4], [3, 5], scope='Conv2d_0b_5x5')
            with tf.variable_scope('Branch_3'):
                branch_3 = slim.avg_pool2d(input_net, [3, 3], scope='AvgPool_0a_3x3')
                branch_3 = slim.conv2d(branch_3, depth_list[5], [1, 1], scope='Conv2d_0b_1x1')
            # add input_net
            output_net = tf.concat([branch_0, branch_1, branch_2, branch_3, input_net], 3)
            return output_net


def inference(input_x,is_training=False):
    #
    net_fuc=inception_v1
    with tf.variable_scope("layer1") :
        # conv1 = slim2.conv2d(input_x, "conv1", 64, [7,7], 2, activation_fn=None)
        conv1 = slim2.conv2d(input_x, "conv1", 64, [5,5], 1 )
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')
        conv2 = slim2.conv2d(pool1, "conv2", 64, [3,3], 1 )
        conv3 = slim2.conv2d(conv2, "conv3", 192, [3,3], 1 )
        pool2 = tf.nn.max_pool(conv3, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')

    # inception modle
    net = net_fuc(pool2,"inception_3a",[64,96,128,16,32,32])
    net = slim2.batch_norm("bn1",net,is_training)
    net = net_fuc(net,"inception_3b",[128,128,192,32,96,64])
    pool3 = tf.nn.max_pool(net, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')
    #
    net = net_fuc(pool3,"inception_4a",[192,96,208,16,48,64])
    net = slim2.batch_norm("bn2",net,is_training)
    net = net_fuc(net,"inception_4b",[160,112,224,24,64,64])
    net = net_fuc(net,"inception_4c",[128,128,256,24,64,64])
    net = net_fuc(net,"inception_4d",[256,160,320,32,64,64])
    pool4 = tf.nn.max_pool(net, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool4')
    #
    net = net_fuc(pool4,"inception_5a",[256,160,320,32,128,128])
    net = net_fuc(net,"inception_5b",[384,192,384,48,128,128])
    pool5 = slim2.global_avg_pool(net)
    # flatten
    resh1 = slim2.flatten(pool5)
    #
    if is_training:
        resh1=tf.nn.dropout(resh1, 0.6)
    output=slim2.fully_connected(resh1, "output", 10, None)
    return output

"""
实验结论：
google, act=relu , sgd=0.5 , 7x7,2, acc=0.782
google, act=relu , sgd=0.5 , 5x5,1, acc=0.827
1，对cifar10数据集，5x5,1 比7x7,2效果好的多
2，加入bn(试验中仅加入2个bn层），acct从0.827提高到 0.866
"""





