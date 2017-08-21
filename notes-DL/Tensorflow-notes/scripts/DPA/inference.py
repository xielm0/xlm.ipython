# -*- coding:utf-8 -*-
import tensorflow as tf

# 配置神经网络的参数
# INPUT_NODE = 29259
OUTPUT_NODE = 1
EMBEDDING_SIZE = 32
LAYER1_NODE = 512
LAYER2_NODE = 256
LAYER3_NODE = 128

N=90
M=75  #多少个连续特征


L1_REGULARAZTION_RATE = 0.05
L2_REGULARAZTION_RATE =0.0003




# inital w
def get_weight_variable(shape, regularizer, network='dnn'):
    if network == 'dnn':
        weights = tf.get_variable("weights", shape, initializer=tf.truncated_normal_initializer(stddev=2.0 / shape[0]),
                                  dtype=tf.float32)
    elif network == 'lr':
        weights = tf.get_variable("weights", shape,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1),
                                  dtype=tf.float32)
    if regularizer != None:
        tf.add_to_collection('losses', regularizer(weights))
    return weights

# input_tensors = [x_continuous, x_category, x_category_onehot]
def inference(input_tensors, train_flag=False ):
    if train_flag:
        regularizer_l1 = tf.contrib.layers.l1_regularizer(L1_REGULARAZTION_RATE)
        regularizer_l2 = tf.contrib.layers.l2_regularizer(L2_REGULARAZTION_RATE)
    else :
        regularizer_l1 = None
        regularizer_l2 = None

    x_continuous, x_category, x_category_onehot = input_tensors
    n_continous_features = x_continuous.shape[1].value
    n_category_features = x_category.shape[1].value

    # 构建DNN网络结构
    with tf.name_scope("embedding"):
        embedding_indexs = range(N-M-1)
        EMBED = []
        # for i in embedding_indexs:
        for i in xrange(n_category_features):
            with tf.variable_scope('embedding_%d' % i):
                if i in embedding_indexs:
                    m = x_category_onehot[i].shape[1].value
                    # EmbedWeights = tf.get_variable("EmbedWeights", shape=[m, EMBEDDING_SIZE],
                    #                              initializer=tf.random_uniform_initializer(-1., 1.))
                    EmbedWeights = tf.get_variable("EmbedWeights", shape=[m, EMBEDDING_SIZE],
                                                 initializer=tf.truncated_normal_initializer(stddev=1))
                    embedi = tf.nn.embedding_lookup(EmbedWeights, tf.cast(x_category[:, i], tf.int32))
                    EMBED.append(embedi)


    with tf.variable_scope('layer_1'):
        # embedding 进行cancat
        embed_output = reduce(lambda a, b: tf.concat([a, b], 1), EMBED)
        input_layer1 = tf.concat([x_continuous, embed_output], 1)
        biases = tf.get_variable("biases", [LAYER1_NODE], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
        weights = get_weight_variable([input_layer1.shape[1].value, LAYER1_NODE], regularizer_l2)
        layer1 = tf.nn.relu(tf.matmul(input_layer1, weights) + biases)
        if train_flag:
            layer1 = tf.nn.dropout(layer1, 0.5)

    with tf.variable_scope('layer_2'):
        weights = get_weight_variable([LAYER1_NODE, LAYER2_NODE], regularizer_l2)
        biases = tf.get_variable("biases", [LAYER2_NODE], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
        layer2 = tf.nn.relu(tf.matmul(layer1, weights) + biases)
        if train_flag:
            layer2 = tf.nn.dropout(layer2, 0.5)

    with tf.variable_scope('layer_3'):
        weights = get_weight_variable([LAYER2_NODE, LAYER3_NODE], regularizer_l2)
        biases = tf.get_variable("biases", [LAYER3_NODE], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
        layer3 = tf.nn.relu(tf.matmul(layer2, weights) + biases)
        if train_flag:
            layer3 = tf.nn.dropout(layer3, 0.5)


    # 构建LR模型
    with tf.variable_scope('logistic-regression'):
        lr_indexs = range(N-M-1)
        vec_features = reduce(lambda a, b: tf.concat([a, b], 1), map(lambda ii: x_category_onehot[ii], lr_indexs))
        lr_input = tf.concat([layer3, vec_features], 1)
        # lr_input = layer3
        weights = get_weight_variable([lr_input.shape[1].value, 1], regularizer_l1, network='lr')

        biases = tf.get_variable("biases", [1], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
        res = tf.nn.sigmoid(tf.matmul(lr_input, weights) + biases)

    return res




