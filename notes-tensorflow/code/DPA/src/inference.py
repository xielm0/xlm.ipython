# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
import processing as pr
import input

# 配置神经网络的参数
OUTPUT_NODE = 1
EMBEDDING_SIZE = 32
LAYER1_NODE = 512
LAYER2_NODE = 256
LAYER3_NODE = 128

N=input.N
M=input.M  #多少个连续特征


# inital w
def get_weight_variable(shape, network='dnn'):
    if network == 'dnn':
        weights = tf.get_variable("weights", shape,
                                  initializer=tf.truncated_normal_initializer(stddev=2.0 / tf.sqrt(float(shape[0]+shape[1]))),
                                  dtype=tf.float32)
    elif network == 'lr':
        weights = tf.get_variable("weights", shape,
                                  initializer=tf.truncated_normal_initializer(stddev=1.0 / tf.sqrt(float(shape[0]))),
                                  dtype=tf.float32)
    return weights


# input_list = [x_continuous, x_category, x_max]
def inference(input_list, train_flag=False ):
    x_continue, x_category, x_max= input_list

    # 对x_category 进行onehot
    n_category_features = len(x_max)
    x_category_onehot=pr.oneHot(x_category,x_max)

    # 构建DNN网络结构
    with tf.name_scope("embedding"):
        embedding_indexs = range(N-M-1)
        EMBED = []
        # for i in embedding_indexs:
        for i in xrange(n_category_features):
            with tf.variable_scope('embedding_%d' % i):
                if i in embedding_indexs:
                    m = x_category_onehot[i].shape[1].value
                    EmbedWeights = tf.get_variable("EmbedWeights", shape=[m, EMBEDDING_SIZE],
                                                 initializer=tf.random_uniform_initializer(-1.0, 1.0))
                    embedi = tf.nn.embedding_lookup(EmbedWeights, tf.cast(x_category[:, i], tf.int32))
                    EMBED.append(embedi)

    with tf.name_scope("layer"):
        with tf.variable_scope('layer_1'):
            # embedding 进行cancat
            embed_output = reduce(lambda a, b: tf.concat([a, b], 1), EMBED)
            input_layer1 = tf.concat([x_continue, embed_output], 1)
            biases = tf.get_variable("biases", [LAYER1_NODE], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
            weights = get_weight_variable([input_layer1.shape[1].value, LAYER1_NODE])
            layer1 = tf.nn.relu(tf.matmul(input_layer1, weights) + biases)
            if train_flag:
                # regularizer = tf.contrib.layers.l2_regularizer(0)
                # tf.add_to_collection('losses', regularizer(weights))
                layer1 = tf.nn.dropout(layer1, 0.8)
            #
            w_sum=tf.reduce_sum(tf.abs(weights))
            tf.summary.scalar("weights",w_sum)

        with tf.variable_scope('layer_2'):
            weights = get_weight_variable([LAYER1_NODE, LAYER2_NODE])
            biases = tf.get_variable("biases", [LAYER2_NODE], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
            layer2 = tf.nn.relu(tf.matmul(layer1, weights) + biases)
            if train_flag:
                # regularizer = tf.contrib.layers.l2_regularizer(0)
                # tf.add_to_collection('losses', regularizer(weights))
                layer2 = tf.nn.dropout(layer2, 0.8)
            #
            w_sum=tf.reduce_sum(tf.abs(weights))
            tf.summary.scalar("weights",w_sum)

        with tf.variable_scope('layer_3'):
            weights = get_weight_variable([LAYER2_NODE, LAYER3_NODE])
            biases = tf.get_variable("biases", [LAYER3_NODE], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
            layer3 = tf.nn.relu(tf.matmul(layer2, weights) + biases)
            if train_flag:
                # regularizer = tf.contrib.layers.l2_regularizer(0)
                # tf.add_to_collection('losses', regularizer(weights))
                layer3 = tf.nn.dropout(layer3, 0.8)
            #
            w_sum=tf.reduce_sum(tf.abs(weights))
            tf.summary.scalar("weights",w_sum)

        # 构建LR模型
        with tf.variable_scope('output'):
            lr_indexs = range(N-M-1)
            vec_features = reduce(lambda a, b: tf.concat([a, b], 1), map(lambda ii: x_category_onehot[ii], lr_indexs))
            lr_input = tf.concat([layer3, vec_features], 1)
            # only dnn, not concat(lr)
            # lr_input = layer3
            weights = get_weight_variable([lr_input.shape[1].value, 1], network='lr')
            biases = tf.get_variable("biases", [1], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
            res = tf.nn.sigmoid(tf.matmul(lr_input, weights) + biases)
            if train_flag:
                regularizer = tf.contrib.layers.l2_regularizer(1e-4)
                tf.add_to_collection('losses', regularizer(weights))
            #
            w_sum=tf.reduce_sum(tf.abs(weights))
            tf.summary.scalar("weights",w_sum)

    return res



