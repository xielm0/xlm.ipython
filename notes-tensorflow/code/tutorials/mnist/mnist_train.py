# -*- coding:utf-8 -*-
import os
import time

import tensorflow as tf
import mnist_inference_cnn as inference
import mnist_input as input


MAX_STEPS = 20000
INITIAL_LEARNING_RATE = 0.001
DECAYG_RATE = 0.99
DECAY_STEPS = 100
MOVING_AVERAGE_DECAY = 0.99

# 定义路径
TENSORBOARD_PATH ="../logs/tensorboard/"
MODEL_SAVE_PATH = "../models/"
MODEL_NAME = "mnist.ckpt"

SESS_CONFIG=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
SESS_CONFIG.gpu_options.allow_growth = True


def get_loss(logits, labels, scope):
    # y_conv = tf.clip_by_value(tf.nn.softmax(logits), 1e-10, 1.0)
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(labels * tf.log(logits), reduction_indices=[1]))
    # cross_entropy_1 = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    # cross_entropy_1 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=tf.argmax(labels,1))
    # cross_entropy = tf.reduce_mean(cross_entropy_1, name='cross_entropy')
    # scope="GPU_i" ,so计算当前GPU上的loss
    # regularization_loss = tf.add_n(tf.get_collection('losses', scope))
    regularization_loss=0
    loss = cross_entropy + regularization_loss

    with tf.name_scope("loss"):
        tf.summary.scalar("cross_entropy", cross_entropy)
        tf.summary.scalar("loss", loss)

    return loss

def main(argv=None):
    # 将简单运算放在CPU上,只有神经网络的训练放在GPU上。
    with tf.device('/cpu:0'):
        # 获取一个batch的数据,x [100, 784]
        x = tf.placeholder(tf.float32, [None, 784])
        y_ = tf.placeholder(tf.float32, [None, 10])
        #
        mnist = input.get_mnist()

    # global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
    # learning_rate = tf.train.exponential_decay(INITIAL_LEARNING_RATE, global_step, DECAY_STEPS, DECAYG_RATE)
    # opt = tf.train.GradientDescentOptimizer(learning_rate)

    i=1
    with tf.device('/gpu:%d' % i):
        with tf.name_scope('GPU_%d' % i) as scope:
            y_conv = inference.inference(x)
            loss = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
            # loss = get_loss(y_conv, y_, scope)
            # train_op = opt.minimize(loss,global_step=global_step)
            opt = tf.train.AdamOptimizer(1e-4)
            train_op = opt.minimize(loss)

    saver = tf.train.Saver()
    with tf.Session(config=SESS_CONFIG) as sess:
        tf.global_variables_initializer().run()
        for step in range(MAX_STEPS):
            batch = mnist.train.next_batch(50)
            if step % 100 == 0:
                _,loss_value = sess.run([train_op,loss],feed_dict={x: batch[0], y_: batch[1]})
                print("step %d, loss = %.10f "%(step, loss_value))
            else:
                _ = sess.run(train_op,feed_dict={x: batch[0], y_: batch[1]})

            if step % 1000 == 0 or (step + 1) == MAX_STEPS:
                checkpoint_path = os.path.join(MODEL_SAVE_PATH, MODEL_NAME)
                saver.save(sess, checkpoint_path, global_step=step)


if __name__ == '__main__':
    tf.app.run()

