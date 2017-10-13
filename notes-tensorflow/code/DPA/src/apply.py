# -*- coding: utf-8 -*-
import time

import tensorflow as tf
import numpy as np

import inference
import train
import input
import processing as pr
import eval
import pandas as pd
import os
from multiprocessing import Process

N=input.N
M=input.M  #多少个连续特征

# CLASSPATH=$(${HADOOP_HDFS_HOME}/bin/hadoop classpath --glob) python eval.py
LOCAL_RESULT_PATH = "../data/result/"
LOCAL_APPLY_PATH = input.LOCAL_APPLY_PATH
N_GPU = 4
BATCH_SIZE = N_GPU*1000
PARALLEL=4

CONFIG = tf.ConfigProto(
    allow_soft_placement=True,
    log_device_placement=False)
CONFIG.gpu_options.allow_growth = True


def evaluate(pid):
    """
    耗时主要分3个部分： read, eval , write .处理一个文件，分别用时为5s,7s,4s
    这里考虑并行执行，通过pid来通知并行度。
    :param pid: 并行id标志。
    :return:
    """
    # build graph
    # 占位符
    x_ = tf.placeholder(tf.float32, [None, N-1], name='x-input')
    batch_ = tf.placeholder(tf.int32, [ ], name='actual_bach_size')

    # 前向传播
    # y = eval.predict_simple(x_, gpu_idx)
    y = eval.predict(x_, batch_, N_GPU)

    os.system('/bin/rm -r ' + LOCAL_RESULT_PATH + '*')

    file_list = input.get_file_list(LOCAL_APPLY_PATH)

    #
    saver = tf.train.Saver()

    with tf.Session(config=CONFIG) as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        # tf.train.get_checkpoint_state 会自动找到目录中最新模型的文件名。
        ckpt = tf.train.get_checkpoint_state(train.MODEL_SAVE_PATH)

        if ckpt and ckpt.model_checkpoint_path:
            # 加载模型，及模型的变量
            saver.restore(sess, ckpt.model_checkpoint_path)

            for idx,file in enumerate(file_list):
                if idx % PARALLEL != pid :
                    continue

                t0 = time.time()
                user_and_sku, X = input.get_apply_data(file)
                t1 = time.time()
                print("read cost %s sec" %(t1-t0))

                # 计算epoch_num
                n = len(user_and_sku)
                epoch_num = n / BATCH_SIZE if n % BATCH_SIZE == 0 else (n / BATCH_SIZE) + 1
                print("n= %s , epoch_num= %s " %(n, epoch_num))

                y_pred = []
                for i in range(epoch_num):
                    # print('loading test data...')
                    a = i* BATCH_SIZE
                    b = (i+1)* BATCH_SIZE
                    b = b if b<=n else n
                    x_batch = X[a:b,]
                    batch_size = b-a
                    y_p = sess.run(y,feed_dict={x_: x_batch, batch_:batch_size})
                    y_pred += y_p.flatten().tolist()
                y_pred = np.array(y_pred)
                print(y_pred.shape)
                t2 = time.time()
                print("eval cost %s sec" %(t2-t1))

                t3=time.time()
                # save
                file_name = file.split("/")[-1]
                result_file = os.path.join(LOCAL_RESULT_PATH, file_name)
                with open(result_file, 'w') as f:
                    for i in range(len(user_and_sku) ):
                        f.writelines(user_and_sku[i, 0] + '\t' + str(user_and_sku[i, 1]) + '\t' + '%f' % y_pred[i] + '\n')
                t4=time.time()
                print("write file cost %s sec" %(t4-t3))

        else:
            print('No checkpoint file found')
        coord.request_stop()
        coord.join(threads)


def main(argv=None):
    # evaluate(0)
    pid_list=range(PARALLEL)
    for i in pid_list:
        p=Process(target=evaluate,args=(i,))
        p.start()


if __name__ == '__main__':
    # tf.app.run()
    main()



