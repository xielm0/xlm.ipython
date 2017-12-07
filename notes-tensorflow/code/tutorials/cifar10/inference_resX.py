# -*- coding: UTF-8 -*-
import tensorflow as tf
import slim as slim2
from inference_res import residual_v2 as res
slim = tf.contrib.slim
tf.logging.set_verbosity(tf.logging.WARN)


def resNext(x, name, groups=32,is_training=False ):
    """这里要求out_depth ==in_depth ， stide=1
    即，这里不做depth和size的变化。
    """
    activation_fn=tf.nn.relu
    with tf.variable_scope(name):
        in_depth = x.get_shape()[-1].value
        out_depth = in_depth
        mid_depth = out_depth /groups
        orig_x=x
        branch=[]

        for i in range(groups):
            with tf.variable_scope('branch_%d'% i):
                x=slim2.bn_conv(x,"sub1",mid_depth,[1,1],1,activation_fn,is_training)
                x=slim2.bn_conv(x,"sub2",mid_depth,[3,3],1,activation_fn,is_training)
                branch.append(x)
        # concat
        x=tf.concat(branch, 3,'concat')
        x=slim2.bn_conv(x,"sub3",out_depth,[3,3],1,activation_fn,is_training)
        x += orig_x

        tf.logging.debug('image after unit %s', x.get_shape())
        return x


def inference(input_x,is_training=False):
    """
    相比resnet,计算速度慢很多。
    要考虑
    """
    block =resNext
    with tf.variable_scope("layer1") :
        net = slim2.conv2d(input_x, "conv1", 64, [5,5], 1, activation_fn=None)
        net = tf.nn.max_pool(net, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')

    # block
    net = res  (net,"block1_1",128,1, is_training,first_block=True)
    net = block(net,"block1_2",4, is_training)
    net = block(net,"block1_3",4, is_training)
    #
    net = res  (net,"block2_1",256,2, is_training)
    net = block(net,"block2_2", 4, is_training)
    net = block(net,"block2_3", 4, is_training)
    net = block(net,"block2_4", 4, is_training)
    net = block(net,"block2_5", 4, is_training)
    net = block(net,"block2_6", 4, is_training)
    #
    net = res  (net,"block3_1",512,2, is_training)
    net = block(net,"block3_2", 4, is_training)
    net = block(net,"block3_3", 4, is_training)
    net = block(net,"block3_4", 4, is_training)
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
论文中，resnext会比res效果好1个百分点。
实践中，由于groups的加入，结构复杂，运行速度慢很多。
acc=0.862

"""





