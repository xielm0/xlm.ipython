# -*- coding:utf-8 -*-
import tensorflow as tf

# 配置神经网络的参数
INPUT_NODE = 784
OUTPUT_NODE = 10
LAYER1_NODE = 500

def inference(input_tensor,regularizer):
    with tf.variable_scope('layer1'):
        weight = tf.get_variable( "weights", [INPUT_NODE, LAYER1_NODE ],
                                   initializer=tf.truncated_normal_initializer(stddev=0.1))

        # 将 regularizer(weights) 加入到 losses集合。
        tf.add_to_collection('losses',regularizer(weight))

        bias = tf.get_variable( "biases", [LAYER1_NODE], initializer=tf.constant_initializer(0.0))

        #
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weight) + bias)

    with tf.variable_scope('layer2'):
        weight = tf.get_variable( "weights", [LAYER1_NODE, OUTPUT_NODE ],
                                         initializer=tf.truncated_normal_initializer(stddev=0.1))
        # 将 regularizer(weights) 加入到 losses集合。
        tf.add_to_collection('losses',regularizer(weight))

        bias = tf.get_variable( "biases", [OUTPUT_NODE], initializer=tf.constant_initializer(0.0))

        #
        layer2 = tf.nn.relu(tf.matmul(layer1, weight) + bias)

    return layer2
