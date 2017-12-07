# -*- coding: UTF-8 -*-
import tensorflow as tf
import slim as slim2
tf.logging.set_verbosity(tf.logging.WARN)


def add_internal_layer(input_x,name,depth,activation_fn=tf.nn.relu,is_training=False):
    with tf.variable_scope(name):
        x=slim2.bn_conv(input_x,"conv1",depth,[1,1],1,activation_fn,is_training)
        x=slim2.bn_conv(x,      "conv2",depth,[3,3],1,activation_fn,is_training)
        y=tf.concat([input_x,x],3)
        return y


def dense_block(x, name, growth_rate,layers_per_block,activation_fn=tf.nn.relu,is_training=False):
    """这里要求out_depth ==in_depth ， stide=1
    即，这里不做depth和size的变化。
    """

    with tf.variable_scope(name):
        for i in range(layers_per_block):
            layer_name="layer_%d"%i
            x=add_internal_layer(x,layer_name,growth_rate,activation_fn,is_training)

        tf.logging.debug('image after unit %s', x.get_shape())
        return x

def transit_layer(x,name,depth,pool_stride,activation_fn=tf.nn.relu,is_training=False):
    """
    通过1x1的卷积核，降低通道数，
    通过pool降低 size
    """
    with tf.variable_scope(name):
        # dense_block的输出是conv ,所以这里先：bn+relu
        x = slim2.batch_norm('bn', x, is_training)
        x = activation_fn(x)
        x = slim2.conv2d(x, 'conv', depth,[1,1], 1, activation_fn=None)
        k=pool_stride
        pool = tf.nn.avg_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='VALID', name='pool')
        # pool = tf.nn.max_pool(x, ksize=[1, k+1, k+1, 1], strides=[1, k, k, 1], padding='SAME', name='pool')
        return pool



def inference(input_x,is_training=False):
    activation_fn=tf.nn.relu
    with tf.variable_scope("layer1") :
        conv1 = slim2.conv2d(input_x, "conv1", 64, [5,5], 1, activation_fn=None)
        net = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')

    # block1
    net = dense_block(net,"block1",32,3,activation_fn, is_training)
    net = transit_layer(net,"transit1",128,2,activation_fn,is_training)
    #
    net = dense_block(net,"block2",32,8,activation_fn, is_training)
    net = transit_layer(net,"transit2",256,2,activation_fn,is_training)
    #
    net = dense_block(net,"block3",32,4,activation_fn, is_training)
    #
    with tf.variable_scope("last") :
        net = slim2.batch_norm('bn_last', net, is_training)
        net = tf.nn.relu(net,"relu_last")
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
1.运行速度远快于resnext，与res几乎相当
2.acc=0.857
"""




