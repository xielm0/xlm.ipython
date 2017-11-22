# -*- coding: UTF-8 -*-
import tensorflow as tf
import slim as slim2
slim = tf.contrib.slim
tf.logging.set_verbosity(tf.logging.WARN)


def block_v1(x, name, groups=32,is_training=False, first_block=False):
    """这里要求out_depth ==in_depth ， stide=1
    即，这里不做depth和size的变化。
    """
    activation_fn=tf.nn.relu
    with tf.variable_scope(name):
        in_depth = x.get_shape()[-1].value
        mid_depth = in_depth /4
        out_depth = in_depth
        orig_x=x
        branch=[]
        for i in range(groups):
            with tf.variable_scope('branch_%d'% i):
                with tf.variable_scope('sub1'):
                    x = slim2.conv2d(x, 'conv1', mid_depth,[1,1], stride=1,activation_fn=activation_fn)
                with tf.variable_scope('sub2'):
                    x = slim2.conv2d(x, 'conv2', mid_depth,[3,3], stride=1, activation_fn=activation_fn)
                    branch.append(x)
        # concat
        with tf.variable_scope('sub3'):
            concat=tf.concat(branch, 3,'concat')
            x=slim2.conv2d(concat, 'conv3', out_depth,[1,1], stride=1,activation_fn=activation_fn)

        x += orig_x

        tf.logging.debug('image after unit %s', x.get_shape())
        return x


def inference(input_x,is_training=False):
    #
    res_fuc =block_v1
    activation_fn =tf.nn.relu
    with tf.variable_scope("layer1") :
        # 在cifar10数据集上，strdie=1的效果要好些。
        net = slim2.conv2d(input_x, "conv1", 128, [5,5], 1, activation_fn=None)
        net = slim2.batch_norm('bn_first', net, is_training)
        net = activation_fn(net,"relu_first")
        net = tf.nn.max_pool(net, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')

    # res1
    net = res_fuc(net,"res1_1",4, is_training)
    net = res_fuc(net,"res1_2",4, is_training)
    net = res_fuc(net,"res1_3",4, is_training)
    #
    with tf.variable_scope("transit1") :
        net = slim2.conv2d(net, 'conv', 256, [1,1], 1, activation_fn=None)
        net = slim2.batch_norm("bn",net,is_training)
        net = activation_fn(net)
        net = tf.nn.max_pool(net, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool')
    #
    net = res_fuc(net,"res2_1",4, is_training)
    net = res_fuc(net,"res2_2",4, is_training)
    net = res_fuc(net,"res2_3",4, is_training)
    #
    with tf.variable_scope("transit2") :
        net = slim2.conv2d(net, 'conv', 512, [1,1], 1, activation_fn=None)
        net = slim2.batch_norm("bn",net,is_training)
        net = activation_fn(net)
        net = tf.nn.max_pool(net, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool')
    #
    net = res_fuc(net,"res3_1",4, is_training)
    net = res_fuc(net,"res3_2",4, is_training)
    net = res_fuc(net,"res3_3",4, is_training)
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
看论文，发现
"""





