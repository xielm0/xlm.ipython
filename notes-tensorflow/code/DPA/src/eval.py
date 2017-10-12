# -*- coding: utf-8 -*-
import time

import tensorflow as tf
from sklearn import metrics
import numpy as np

import inference
import train
import input
import processing as pr

N=input.N
M=input.M  #多少个连续特征

# CLASSPATH=$(${HADOOP_HDFS_HOME}/bin/hadoop classpath --glob) python eval.py
BATCH_SIZE = 1000*4
STEPS = 200
EVAL_INTERVAL_SECS = 60

CONFIG = tf.ConfigProto(
    allow_soft_placement=True,
    log_device_placement=False)
CONFIG.gpu_options.allow_growth = True

N_GPU=4


def predict_simple(x_, gpu_idx):
    """
    用1个gpu去predict
    :param x_: 输入特征, tensor
    :param gpu_idx:
    :return:
    """
    with tf.device('/gpu:%d' % gpu_idx):
        # 数据处理
        x_list= pr.input_fn(x_, "apply")

        # 前向传播
        y = inference.inference(x_list, False)
    return y


def predict(x_, batch_size, n_gpu):
    """
    用n个gpu去运行。这里只是构建了graph，这里的for循环只是构建了分支，在sess.run的时候是并行执行的。
    :param x:输入特征, tensor
    :return: y - tensor
    """
    y_list = []
    batch_each= batch_size/n_gpu
    for i in range(n_gpu):
        with tf.device('/gpu:%d' % i):
            with tf.name_scope('GPU_%d' % i) as scope:
                c=i*batch_each
                d=(i+1)* batch_each if i < n_gpu-1 else batch_size
                x_i = x_[c:d, ]
                # 数据处理
                x_list= pr.input_fn(x_i,"apply")
                # 前向传播
                y_i = inference.inference(x_list, False)
                tf.get_variable_scope()._reuse=True
                y_list.append(y_i)

    tf.get_variable_scope()._reuse=False
    y = tf.concat(y_list,0)

    return y


def evaluate():
    # build graph
    # 占位符
    x_ = tf.placeholder(tf.float32, [None, N-1], name='x-input')
    y_ = tf.placeholder(tf.int64, [None, ], name='y-input')
    batch_ = tf.placeholder(tf.int32, [ ], name='actual_bach_size')

    # y = predict_simple(x_,0)
    y = predict(x_, batch_, N_GPU)

    # 获取测试数据
    X, Y = input.get_test_data()

    # 计算epoch_num
    n = len(X)
    epoch_num = n / BATCH_SIZE if n % BATCH_SIZE == 0 else (n / BATCH_SIZE) + 1
    print("n= %s , epoch_num= %s " %(n,epoch_num))
    #
    saver = tf.train.Saver()

    with tf.Session(config=CONFIG) as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        while True:
            # tf.train.get_checkpoint_state 会自动找到目录中最新模型的文件名。
            ckpt = tf.train.get_checkpoint_state(train.MODEL_SAVE_PATH)

            if ckpt and ckpt.model_checkpoint_path:
                # 加载模型，及模型的变量
                saver.restore(sess, ckpt.model_checkpoint_path)
                # 通过文件名获得模型保存时的迭代的轮数
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            else:
                print('No checkpoint file found')

            t1 = time.time()
            # 评估，这里是AUC评估。使用sk-learn的metrics
            y_fact = []
            y_pred = []

            for i in range(epoch_num):
                # print('loading test data...')
                a = i* BATCH_SIZE
                b = (i+1)* BATCH_SIZE
                b = b if b<=n else n
                x_batch = X[a:b,]
                y_batch = Y[a:b,]
                batch_size = b-a
                y_f, y_p = sess.run([y_, y], feed_dict={x_: x_batch, y_: y_batch, batch_: batch_size})
                y_fact += y_f.tolist()
                y_pred += y_p.flatten().tolist()
            y_fact = np.array(y_fact)
            y_pred = np.array(y_pred)
            print(y_pred.shape)
            t2 = time.time()
            print("eval cost %s sec" %(t2-t1))

            fpr, tpr, thresholds = metrics.roc_curve(y_fact, y_pred)
            auc_scroe = metrics.auc(fpr, tpr)
            print("After %s training step(s), AUC score = %g" % (global_step, auc_scroe))
            time.sleep(EVAL_INTERVAL_SECS)
    coord.request_stop()
    coord.join(threads)


def main(argv=None):
    evaluate()

if __name__ == '__main__':
    tf.app.run()

