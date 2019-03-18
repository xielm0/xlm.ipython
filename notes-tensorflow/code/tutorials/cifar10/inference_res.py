# -*- coding: UTF-8 -*-
import tensorflow as tf
import slim as slim2
slim = tf.contrib.slim
tf.logging.set_verbosity(tf.logging.WARN)


def residual(x, name, out_depth, stride=1, is_training=False,first_block=False):
    """Residual unit with 2 sub layers."""
    activation_fn=tf.nn.relu
    with tf.variable_scope(name):
        in_depth = x.get_shape()[-1].value
        with tf.variable_scope('sub1'):
            if first_block:
                x = slim2.batch_norm('bn1', x, is_training)
                x = activation_fn(x)
                orig_x=x
                x = slim2.conv2d(x, 'conv1', out_depth,[3,3], stride,activation_fn=None)
            else:
                orig_x=x
                x = slim2.batch_norm('bn1', x, is_training)
                x = activation_fn(x)
                x = slim2.conv2d(x, 'conv1', out_depth,[3,3], stride,activation_fn=None)

        with tf.variable_scope('sub2'):
            x = slim2.batch_norm('bn2', x, is_training)
            x = activation_fn(x)
            x = slim2.conv2d(x, 'conv2', out_depth,[3,3], stride=1,activation_fn=None)

        with tf.variable_scope('sub_add'):
            if in_depth != out_depth:
                shortcut = tf.nn.avg_pool(orig_x, [1,stride,stride,1], [1,stride,stride,1], 'SAME')
                shortcut = tf.pad(shortcut, [[0, 0], [0, 0], [0, 0],
                                         [(out_depth-in_depth)//2, (out_depth-in_depth)//2]])
            x += shortcut

        tf.logging.debug('image after unit %s', x.get_shape())
        return x


def residual_v2(x, name, out_depth, stride=1, is_training=False, first_block=False):
    """bn->relu->conv , 将conv的结果进行shortcut"""
    activation_fn=tf.nn.relu
    with tf.variable_scope(name):
        mid_depth=out_depth/4
        in_depth = x.get_shape()[-1].value
        with tf.variable_scope('sub1'):
            if first_block:
                x = slim2.batch_norm('bn1', x, is_training)
                x = activation_fn(x)
                orig_x=x
                x = slim2.conv2d(x, 'conv1', mid_depth,[1,1], stride=1,activation_fn=None)
            else:
                orig_x=x
                x = slim2.batch_norm('bn1', x, is_training)
                x = activation_fn(x)
                x = slim2.conv2d(x, 'conv1', mid_depth,[1,1], stride=1,activation_fn=None)

        with tf.variable_scope('sub2'):
            x = slim2.batch_norm('bn2', x, is_training)
            x = activation_fn(x)
            x = slim2.conv2d(x, 'conv2', mid_depth,[3,3], stride=stride, activation_fn=None)

        with tf.variable_scope('sub3'):
            x = slim2.batch_norm('bn3', x, is_training)
            x = activation_fn(x)
            x = slim2.conv2d(x, 'conv3', out_depth,[1,1], stride=1,activation_fn=None)

        with tf.variable_scope('sub_add'):
            if in_depth != out_depth:
                shortcut = slim2.conv2d(orig_x, 'shortcut', out_depth, [stride,stride], stride, activation_fn=None)
            else:
                shortcut = slim2.subsample(orig_x, stride)
            x += shortcut

        tf.logging.debug('image after unit %s', x.get_shape())
        return x


def residual_v3(x, name, out_depth, stride=1, is_training=False):
    """conv ->bn -> relu ,将relu后的结果进行shortcut"""
    activation_fn=tf.nn.relu
    with tf.variable_scope(name):
        mid_depth=out_depth/4
        in_depth = x.get_shape()[-1].value
        with tf.variable_scope('sub1'):
            orig_x=x
            x = slim2.conv2d(x, 'conv1', mid_depth,[1,1], stride=1,activation_fn=None)
            x = slim2.batch_norm('bn1', x, is_training)
            x = activation_fn(x)

        with tf.variable_scope('sub2'):
            x = slim2.conv2d(x, 'conv2', mid_depth,[3,3], stride=stride, activation_fn=None)
            x = slim2.batch_norm('bn2', x, is_training)
            x = activation_fn(x)

        with tf.variable_scope('sub3'):
            x = slim2.conv2d(x, 'conv3', out_depth,[1,1], stride=1,activation_fn=None)
            x = slim2.batch_norm('bn3', x, is_training)

        with tf.variable_scope('sub_add'):
            if in_depth != out_depth:
                shortcut = slim2.conv2d(orig_x, 'shortcut', out_depth, [stride,stride], stride, activation_fn=None)
                shortcut = slim2.batch_norm('bn4', shortcut, is_training)
            else:
                shortcut = slim2.subsample(orig_x, stride)
            x += shortcut
            x = activation_fn(x)

        tf.logging.debug('image after unit %s', x.get_shape())
        return x


def inference(input_x,is_training=False):
    """
    50-layer,
    论文中input size = 224 x 224 ,而cifar10的输入是24 x24
    """
    res_fuc =residual_v2
    with tf.variable_scope("layer1") :
        net = slim2.conv2d(input_x, "conv1", 64, [5,5], 1, activation_fn=None)
        # net = slim2.batch_norm('bn_first', net, is_training)
        # net = tf.nn.relu(net,"relu_first")
        net = tf.nn.max_pool(net, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')
    # res1
    net = res_fuc(net,"res64_1",128,1, is_training,first_block=True)
    net = res_fuc(net,"res64_2",128,1, is_training)
    net = res_fuc(net,"res64_3",128,1, is_training)
    #
    net = res_fuc(net,"res128_1",256,2, is_training)
    net = res_fuc(net,"res128_2",256,1, is_training)
    net = res_fuc(net,"res128_3",256,1, is_training)
    net = res_fuc(net,"res128_4",256,1, is_training)
    #
    net = res_fuc(net,"res256_1",512,2, is_training)
    net = res_fuc(net,"res256_2",512,1, is_training)
    net = res_fuc(net,"res256_3",512,1, is_training)
    net = res_fuc(net,"res256_4",512,1, is_training)
    # net = res_fuc(net,"res256_5",512,1, is_training)
    # net = res_fuc(net,"res256_6",512,1, is_training)
    # #
    # net = res_fuc(net,"res512_1",1024,2, is_training)
    # net = res_fuc(net,"res512_2",1024,1, is_training)
    # net = res_fuc(net,"res512_3",1024,1, is_training)
    #
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
resnet_model，DBA采用的是bn + relu + conv ,是因为
1，DBA的上一层是conv or conv+short cut,而不是 conv + bn +relu,
2，short cut = conv = origx_x,而不是shortcut=relu
这里也设计了res-v3,即采用conv+bn+relu , 即shortcut=relu，训练速度明显慢很多，acc也差
实验结论：
depth=50-layer/4
res , act=relu , sgd=0.5 , bn=None , acc=0.821
res , act=relu , sgd=0.5 , first_block=True , acc=0.853 --比alexnet差的原因是pool层过多
res , act=relu , sgd=0.5 , first_block=False, acc=0.853
res-v3 , acc=0.789
depth=50-layer
res , act=relu , sgd=0.5 , first_block=True , acc=0.851
"""




