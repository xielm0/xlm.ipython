# -*- coding: UTF-8 -*-
import tensorflow as tf
import numpy as np
import os
import time
import input
import inference
from slim import BN_OPS

DATA_DIR = '../data/cifar10/cifar-10-batches-bin'
BATCH_SIZE = 128
MAX_STEPS = 20000
MOVING_AVERAGE_DECAY = 0.99

# 定义日志和模型输出的路径
TENSORBOARD_PATH ="../logs/tensorboard/"
CKPT_DIR = "../models/cifar"
CKPT_NAME = "cifar.ckpt"

if not os.path.exists(CKPT_DIR):
    os.mkdir(CKPT_DIR)

CONFIG = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
CONFIG.gpu_options.allow_growth = True


def get_loss(x, labels):
    logits = inference.inference(x, True)
    cross_entropy_1 = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=labels, name='cross_entropy_per_example')
    cross_entropy = tf.reduce_mean(cross_entropy_1, name='cross_entropy')
    # scope="GPU_i" ,so计算当前GPU上的loss
    # regularization_loss = tf.add_n(tf.get_collection('losses'))
    regularization_loss=0
    loss = cross_entropy + regularization_loss

    with tf.name_scope("loss"):
        tf.summary.scalar("cross_entropy", cross_entropy)
        tf.summary.scalar("loss", loss)

    return loss


def build_model(image_holder, label_holder):
    loss = get_loss(image_holder, label_holder)
    #
    global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
    learning_rate = tf.train.exponential_decay(learning_rate=0.5, global_step=global_step,
                                               decay_steps=100, decay_rate=0.98)
    tf.summary.scalar('learning-rate', learning_rate)
    opt = tf.train.GradientDescentOptimizer(learning_rate)
    # opt = tf.train.AdamOptimizer(1e-4)
    apply_gradient_op = opt.minimize(loss, global_step=global_step)

    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    # train_op = tf.group(apply_gradient_op,variables_averages_op)
    # add batch_nrom operation
    train_op = tf.group(apply_gradient_op,variables_averages_op, *BN_OPS)

    return loss, train_op


def train():
    with tf.device('/cpu:0'):
        images_train, labels_train = input.distorted_inputs(data_dir=DATA_DIR, batch_size=BATCH_SIZE)

    # image_holder = tf.placeholder(tf.float32, [BATCH_SIZE, 24, 24, 3])
    # label_holder = tf.placeholder(tf.int32, [BATCH_SIZE])
    image_holder = images_train
    label_holder = labels_train

    loss, train_op = build_model(image_holder, label_holder)

    # tensorboard
    cmd = "/bin/rm  " + TENSORBOARD_PATH + "events.out.tfevents.*"
    os.system(cmd)
    merged_summary_op = tf.summary.merge_all()

    #
    saver = tf.train.Saver()
    with tf.Session(config=CONFIG) as sess:
        # init
        tf.global_variables_initializer().run()
        #
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        #
        summary_writer = tf.summary.FileWriter(TENSORBOARD_PATH, sess.graph)
        for step in range(MAX_STEPS):
            if step != 0 and step % 100 == 0:
                #
                t1 = time.time()
                _, loss_value = sess.run([train_op, loss])
                #
                duration = time.time() - t1
                examples_per_sec = BATCH_SIZE / duration
                sec_per_batch = duration
                format_str = ('step %d, loss = %.2f (%.1f examples/sec; %.3f sec/batch)')
                print(format_str % (step, loss_value, examples_per_sec, sec_per_batch))
                #
                summary = sess.run(merged_summary_op)
                summary_writer.add_summary(summary, step)

            else:
                # 仅仅执行
                _, loss_value = sess.run([train_op, loss])

            if step % 1000 == 0 or (step + 1) == MAX_STEPS:
                checkpoint_path = os.path.join(CKPT_DIR, CKPT_NAME)
                saver.save(sess, checkpoint_path, global_step=step)

        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    train()
