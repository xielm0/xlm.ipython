# -*- coding: UTF-8 -*-
import tensorflow as tf
from tensorflow.python.training import moving_averages
import numpy as np

x2y = lambda x: x
swish = lambda z: z*tf.nn.sigmoid(z)
BN_OPS = []


def leaky_relu(x, leakiness=0.0):
    """Relu, with optional leaky support."""
    return tf.where(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')


def conv(input_op, name, depth, ksize, stride ):
    with tf.variable_scope(name) as scope:
        n_in = input_op.get_shape()[-1].value
        [kh,kw]=ksize
        weights_initializer=tf.contrib.layers.xavier_initializer_conv2d()
        # weights_initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0/(kh*kw*depth)))
        kernel = tf.get_variable("kernel",
                                 shape=[kh, kw, n_in, depth],
                                 dtype=tf.float32,
                                 initializer=weights_initializer)
        conv = tf.nn.conv2d(input_op, kernel, strides=[1,stride,stride,1], padding='SAME')
        return conv


def conv2d(input_op, name, depth, ksize, stride, activation_fn=tf.nn.relu):
    if activation_fn == None:
        activation_fn=x2y
    with tf.variable_scope(name) as scope:
        n_in = input_op.get_shape()[-1].value
        [kh,kw]=ksize
        weights_initializer=tf.contrib.layers.xavier_initializer_conv2d()
        # weights_initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0/(kh*kw*depth)))
        kernel = tf.get_variable("kernel",
                                 shape=[kh, kw, n_in, depth],
                                 dtype=tf.float32,
                                 initializer=weights_initializer)
        conv = tf.nn.conv2d(input_op, kernel, strides=[1,stride,stride,1], padding='SAME')
        biases = tf.get_variable('biases', shape=[depth], initializer=tf.constant_initializer(0),  dtype=tf.float32)
        z = tf.nn.bias_add(conv, biases)
        out = activation_fn(z)
        return out


def fully_connected(input_op, name, n_out, activation_fn=None):
    if activation_fn == None:
        activation_fn=x2y
    with tf.variable_scope(name) as scope:
        n_in = input_op.get_shape()[-1].value
        # weights_initializer =tf.contrib.layers.xavier_initializer()
        weights_initializer =tf.truncated_normal_initializer(stddev=1.0 / tf.sqrt(float(n_in)))
        weights = tf.get_variable("weights",
                                  shape=[n_in, n_out],
                                  dtype=tf.float32,
                                  initializer=weights_initializer)
        biases = tf.get_variable('biases', shape=[n_out], initializer=tf.constant_initializer(0),  dtype=tf.float32)
        z=tf.matmul(input_op, weights) + biases
        out=activation_fn(z)
        return out


def global_avg_pool(x):
    assert x.get_shape().ndims == 4
    return tf.reduce_mean(x, [1, 2])


def flatten(x):
    x_shape = x.get_shape()
    x_rank = len(x_shape) # x.get_shape().ndims
    if (x_rank is None) or (x_rank < 2):
        raise ValueError('Inputs must have a least 2 dimensions.')
    shp = x_shape.as_list()
    flattened_shape = np.prod(shp[1:x_rank])
    resh1 = tf.reshape(x, [shp[0], flattened_shape])
    return resh1


# slim.batch_norm(inputs, activation_fn=None, is_training=False,scope='bn1')
def batch_norm(name, x, is_training=False):
    bn_decay=0.99
    x_shape = x.get_shape()
    x_rank = len(x_shape)
    axis = list(range(x_rank - 1))

    with tf.variable_scope(name):
        params_shape = [x_shape[-1]]
        beta = tf.get_variable('beta', params_shape, initializer=tf.zeros_initializer())
        gamma = tf.get_variable('gamma', params_shape, initializer=tf.ones_initializer())

        moving_mean = tf.get_variable('moving_mean', params_shape, initializer=tf.zeros_initializer(), trainable=False)
        moving_variance = tf.get_variable('moving_variance', params_shape, initializer=tf.ones_initializer(), trainable=False)

        if is_training:
            mean, variance = tf.nn.moments(x, axis)
            update_moving_mean = moving_averages.assign_moving_average(moving_mean, mean, bn_decay)
            update_moving_variance = moving_averages.assign_moving_average(moving_variance, variance, bn_decay)
            BN_OPS.append(update_moving_mean)
            BN_OPS.append(update_moving_variance)
            # need sess.run(BN_OPS)
        else:
            mean, variance = moving_mean, moving_variance

    return tf.nn.batch_normalization(x, mean, variance, beta, gamma, 1e-8)


def subsample(inputs, stride, scope=None):
    """如 6x6 ,变成3x3"""
    if stride == 1:
        return inputs
    else:
        # return slim.max_pool2d(inputs, [1, 1], stride=stride, scope=scope)
        return tf.nn.max_pool(inputs, ksize=[1, stride, stride, 1], strides=[1, stride, stride, 1], padding='SAME', name=scope)


def shortcut(x,name,out_depth,stride=1, activation_fn=None):
    with tf.variable_scope(name):
        in_depth = x.get_shape()[-1].value
        if in_depth != out_depth:
            shortcut = conv2d(x, 'conv1x1', out_depth, [stride,stride], stride, activation_fn)
        else:
            shortcut = subsample(x, stride)
        return shortcut



def bn_conv(x,name,depth, ksize=[3,3],stride=1,activation_fn=tf.nn.relu,is_training=False):
    with tf.variable_scope(name):
        x = batch_norm('bn', x, is_training)
        x = activation_fn(x)
        x = conv2d(x, 'conv', depth,ksize, stride=stride, activation_fn=None)
        return x



