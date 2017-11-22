# -*- coding: utf-8 -*-
import time

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_inference_cnn as inference
import mnist_train as train
import numpy as np


def evaluate():
    mnist = input_data.read_data_sets("/export/Data/MNIST_data", one_hot=True)

    x = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder(tf.float32, [None, 10])
    validated_feed = {x: mnist.test.images, y_: mnist.test.labels}
    # m=mnist.train.next_batch(10000)
    # validated_feed ={x:m[0],y_:m[1]}

    # 前向传播
    y = inference.inference(x, False)

    #计算正确率
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # saver 用来加载模型
    # ema=tf.train.ExponentialMovingAverage(train.MOVING_AVERAGE_DECAY)
    # saver = tf.train.Saver(ema.variables_to_restore())
    saver = tf.train.Saver()

    while True:
        with tf.Session() as sess:
            # tf.train.get_checkpoint_state 会自动找到目录中最新模型的文件名。
            ckpt = tf.train.get_checkpoint_state(train.MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                # 加载模型
                saver.restore(sess, ckpt.model_checkpoint_path)
                # 通过文件名获得模型保存时的迭代的轮数
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                # y_real = sess.run(y ,feed_dict=validated_feed)
                # print(np.argmax(y_real[:100],1))
                # print(np.argmax(mnist.test.labels[:100],1))
                # 评估指标
                accuracy_score = sess.run(accuracy, feed_dict=validated_feed)
                print("After %s training step(s), validation accuracy = %g" % (global_step, accuracy_score))
            else:
                print('No checkpoint file found')
                return
        time.sleep(10)

def main(argv=None):
    evaluate()

if __name__ == '__main__':
    tf.app.run()



