# -*- coding: UTF-8 -*-
import tensorflow as tf
import slim as sl
NUM_CLASSES=10

def inference_le(x_image,is_training=False):
    # activation_fn=tf.nn.relu
    activation_fn =sl.swish
    #
    conv1 = sl.conv2d(x_image,"conv1", depth=32,ksize=[5, 5], stride=1, activation_fn=activation_fn)
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],padding='SAME', name='pool1')
    conv2 = sl.conv2d(pool1,"conv2", depth=64,ksize=[5, 5], stride=1, activation_fn=activation_fn)
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')
    #
    resh1 = sl.flatten(pool2)
    #
    fc1=sl.fully_connected(resh1,name="fc1",n_out=512, activation_fn=activation_fn)
    if is_training:
        fc1=tf.nn.dropout(fc1, 0.5)

    fc2=sl.fully_connected(fc1,name="fc2",n_out=256, activation_fn=activation_fn)
    if is_training:
        fc2=tf.nn.dropout(fc2, 0.5)

    output=sl.fully_connected(fc2,"output", NUM_CLASSES, activation_fn=None)
    return output


def inference_alex(x_image,is_training=False):
    activation_fn=tf.nn.relu
    # activation_fn =sl.swish
    #
    # conv1 = sl.conv_op(x_image,"conv1",96, [11, 11], 4,activation_fn)
    # ksize=[11,11],stride=4, 这对提高acc非常不利，改为5x5的filter,stride=1,效果要好很多。
    conv1 = sl.conv2d(x_image,"conv1",96, [5, 5], 1,activation_fn)
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')
    conv2 = sl.conv2d(pool1,"conv2",256, [5, 5], 1,activation_fn)
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')

    conv3 = sl.conv2d(pool2,"conv3",384,[3, 3], 1,activation_fn)
    conv4 = sl.conv2d(conv3,"conv4",384,[3, 3], 1,activation_fn)
    conv5 = sl.conv2d(conv4,"conv5",256,[3, 3], 1,activation_fn)
    pool3 = tf.nn.max_pool(conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')
    # flatten
    resh1 = sl.flatten(pool3)
    #
    fc1=sl.fully_connected(resh1,name="fc1",n_out=512, activation_fn=activation_fn)
    if is_training:
        fc1=tf.nn.dropout(fc1, 0.5)

    fc2=sl.fully_connected(fc1,name="fc2",n_out=256, activation_fn=activation_fn)
    if is_training:
        fc2=tf.nn.dropout(fc2, 0.5)

    output=sl.fully_connected(fc2,"output", NUM_CLASSES, activation_fn=None)
    return output


# inference = inference_alex
import inference_res as inference
inference = inference.inference

"""
实验数据：
1. lenet, act=relu ,  sgd=0.5, acc=0.776
   lenet, act=swish,  sgd=0.5, acc=0.816
2. alex , act=relu,   sgd=0.5, acc=0.730
   alex,  act=relu ,  sgd=0.5, 5x5,1, acc=0.843
   alex,  act=swish,  sgd=0.5, acc=0.747
   alex,  act=swish,  sgd=0.5, 5x5,1, acc=0.869
3, google, act=relu , sgd=0.5, 5x5,1, acc=0.827
   google, act=relu , sgd=0.5, 5x5,1,bn, acc=0.866
4, res , act=relu , sgd=0.5 , first_block=True , acc=0.853
5, resx,
6，xlm, act=relue , sgd=0.5,  stride=1,0.851
"""
