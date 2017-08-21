# -*- coding:utf-8 -*-
import tensorflow as tf

# 配置神经网络的参数
INPUT_NODE = 784
OUTPUT_NODE = 10

IMAGE_SIZE = 28
NUM_CHANNELS = 1
NUM_LABELS = 10

# 第一层卷积层的尺寸和深度
CONV1_DEEP = 32
CONV1_SIZE = 5
# 第二层卷积层的尺寸和深度
CONV2_DEEP = 64
CONV2_SIZE = 5
# 全连接层的节点个数
FC_SIZE = 512



# 生成变量监控信息
def variable_summaries(var,name):
    with tf.name_scope("summaries"):
        tf.summary.histogram(name, var)
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean/' + name, mean)
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev/' + name, stddev)

def inference(input_tensor,drop_out_flag,  regularizer):
    with tf.variable_scope('layer1-conv1'):
        weight = tf.get_variable(
            "weight", [CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_DEEP],
            initializer=tf.truncated_normal_initializer(stddev=0.1))
        # variable_summaries(conv1_weights)

        bias = tf.get_variable(
            "bias", [CONV1_DEEP], initializer=tf.constant_initializer(0.0))
        # variable_summaries(conv1_biases)

        conv = tf.nn.conv2d(
            input_tensor, weight, strides=[1, 1, 1, 1], padding='SAME')
        # tf.summary.histogram("layer1-conv1/pre_activations", conv1 + conv1_biases)

        relu1 = tf.nn.relu(tf.nn.bias_add(conv, bias))
        # tf.summary.histogram("layer1-conv1/activations", relu1)

    with tf.name_scope('layer2-pool1'):
        pool1 = tf.nn.max_pool(
            relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        # tf.summary.histogram("layer2-pool1/summaries/pool1", pool1)

    with tf.variable_scope('layer3-conv2'):
        weight = tf.get_variable(
            "weight", [CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP],
            initializer=tf.truncated_normal_initializer(stddev=0.1))
        bias = tf.get_variable(
            "bias", [CONV2_DEEP],
            initializer=tf.constant_initializer(0.0))
        conv = tf.nn.conv2d(
            pool1, weight, strides=[1, 1, 1, 1], padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv, bias))

    with tf.name_scope('layer4-pool2'):
        pool2 = tf.nn.max_pool(
            relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    pool_shape = pool2.get_shape().as_list()
    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]

    reshaped = tf.reshape(pool2, [-1, nodes])

    with tf.variable_scope('layer5-fc1'):
        weight = tf.get_variable(
            "weight", [nodes, FC_SIZE],
            initializer=tf.truncated_normal_initializer(stddev=0.1))
        tf.add_to_collection("losses", regularizer(weight))

        bias = tf.get_variable(
            "bias", [FC_SIZE], initializer=tf.constant_initializer(0.1))
        fc1 = tf.nn.relu(tf.matmul(reshaped, weight) + bias)

        if drop_out_flag:
            fc1 = tf.nn.dropout(fc1, 0.5)
    with tf.variable_scope('layer6-fc2'):
        weight = tf.get_variable(
            "weight", [FC_SIZE, NUM_LABELS],
            initializer=tf.truncated_normal_initializer(stddev=0.1))
        tf.add_to_collection("losses", regularizer(weight))
        bias = tf.get_variable(
            "bias", [NUM_LABELS],
            initializer=tf.constant_initializer(0.1))
        fc2 = tf.matmul(fc1, weight) + bias

    return fc2
