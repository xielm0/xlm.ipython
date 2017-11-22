# -*- coding: UTF-8 -*-
import tensorflow as tf
import numpy as np
import time
import math
import input
import inference
import train

BATCH_SIZE = 128
DATA_DIR = '../data/cifar10/cifar-10-batches-bin'
CKPT_DIR=train.CKPT_DIR

CONFIG = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
CONFIG.gpu_options.allow_growth = True


def eval():
    with tf.device('/cpu:2'):
        images_test, labels_test = input.inputs(eval_data=True,
                                                data_dir=DATA_DIR,
                                                batch_size=BATCH_SIZE)

    image_holder = images_test
    label_holder = labels_test

    # predict
    with tf.device('/gpu:%d' % 2):
        logits = inference.inference(image_holder)

        # tf.nn.in_top_k(predictions, targets, k ) 判断target 是否在top k 的预测之中。
        top_k_op = tf.nn.in_top_k(logits, label_holder, 1)

    ema=tf.train.ExponentialMovingAverage(train.MOVING_AVERAGE_DECAY)
    saver = tf.train.Saver(ema.variables_to_restore())
    # saver = tf.train.Saver()
    with tf.Session(config=CONFIG) as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        #
        num_examples = 20000
        num_iter = int(math.ceil(num_examples / BATCH_SIZE))
        total_sample_count = num_iter * BATCH_SIZE

        while True:
            ckpt = tf.train.get_checkpoint_state(CKPT_DIR)
            if ckpt and ckpt.model_checkpoint_path:
                # 加载模型，及模型的变量
                saver.restore(sess, ckpt.model_checkpoint_path)
                # 通过文件名获得模型保存时的迭代的轮数
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            else:
                print('No checkpoint file found')
                return

            step = 0
            true_count = 0
            while step < num_iter and not coord.should_stop():
                predictions = sess.run([top_k_op])
                true_count += np.sum(predictions)
                step += 1
            precision = 1.0 * true_count / total_sample_count
            tf.logging.debug("true_count=%s,total_sample_count= %d" %(true_count,total_sample_count))
            print('After %s training steps,precision @ 1 = %.3f' % (global_step, precision))
            time.sleep(5)

        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    eval()



