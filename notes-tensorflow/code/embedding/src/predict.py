# -*- coding: utf-8 -*-
import time
import tensorflow as tf
import numpy as np
import train
import input
import pandas as pd
import os
from multiprocessing import Process


# CLASSPATH=$(${HADOOP_HDFS_HOME}/bin/hadoop classpath --glob) python eval.py
LOCAL_APPLY_PATH =input.LOCAL_APPLY_PATH
LOCAL_RESULT_PATH = "../data/result/"
N_GPU = 4
BATCH_SIZE = N_GPU*1000

CONFIG = tf.ConfigProto(
    allow_soft_placement=True,
    log_device_placement=False)
CONFIG.gpu_options.allow_growth = True

cid_list=train.cid_list
run_cid_list=train.run_cid_list

if not os.path.exists(LOCAL_RESULT_PATH):
    os.mkdir(LOCAL_RESULT_PATH)
for cid in cid_list:
    path = os.path.join(LOCAL_RESULT_PATH,cid)
    if not os.path.exists(path):
        os.mkdir(path)

discrete_nums=train.discrete_nums
vocabulary_size=train.vocabulary_size
embedding_size=train.embedding_size
embedding_size_list=train.embedding_size_list

def inference(train_inputs,cid):
    #
    x_max  = input.get_max(cid)

    #
    embed_list = []
    for i in range(discrete_nums):
        with tf.variable_scope('embedding_%d' % i):
            m = x_max[i]+1
            EmbedWeights = tf.get_variable("EmbedWeights", shape=[m, embedding_size_list[i]],
                                           initializer=tf.random_uniform_initializer(-1.0, 1.0))
            embed_i = tf.nn.embedding_lookup(EmbedWeights, tf.cast(train_inputs[:, i], tf.int32))
            embed_list.append(embed_i)
    embed = reduce(lambda a, b: tf.concat([a, b], 1), embed_list)

    return embed



def evaluate(cid):
    # build graph
    # 占位符
    x_ = tf.placeholder(tf.int32, shape=[None,discrete_nums], name='x-input')
    batch_ = tf.placeholder(tf.int32, [ ], name='actual_bach_size')

    # 前向传播
    i= run_cid_list.index(cid) % 2
    with tf.device('/gpu:%d'% i ):
        y = inference(x_ ,cid)

    os.system('/bin/rm -r %s/*' % ( os.path.join(LOCAL_RESULT_PATH , cid)))
    file_list = input.get_file_list(os.path.join(LOCAL_APPLY_PATH , cid))

    #
    saver = tf.train.Saver()

    with tf.Session(config=CONFIG) as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        # tf.train.get_checkpoint_state 会自动找到目录中最新模型的文件名。
        model_save_path =os.path.join(train.MODEL_SAVE_PATH,cid)
        ckpt = tf.train.get_checkpoint_state(model_save_path)

        if ckpt and ckpt.model_checkpoint_path:
            # 加载模型，及模型的变量
            saver.restore(sess, ckpt.model_checkpoint_path)

            for idx,file in enumerate(file_list):
                sku_id, X = input.get_apply_data(file)

                # 计算epoch_num
                n = len(sku_id)
                epoch_num = n / BATCH_SIZE if n % BATCH_SIZE == 0 else (n / BATCH_SIZE) + 1
                print("n= %s , epoch_num= %s " %(n, epoch_num))

                t1 = time.time()
                y_pred = []
                for i in range(epoch_num):
                    # print('loading test data...')
                    a = i* BATCH_SIZE
                    b = (i+1)* BATCH_SIZE
                    b = b if b<=n else n
                    x_batch = X[a:b,]
                    batch_size = b-a
                    y_p = sess.run(y,feed_dict={x_: x_batch, batch_:batch_size})
                    y_pred += y_p.tolist()
                y_pred = np.array(y_pred)
                print(y_pred.shape)
                t2 = time.time()
                print("eval cost %s sec" %(t2-t1))

                t1=time.time()
                # save
                file_name = file.split("/")[-1]
                result_file = os.path.join(LOCAL_RESULT_PATH, cid, file_name)
                with open(result_file, 'w') as f:
                    for i in range(len(sku_id) ):
                        s=map(lambda x: str(x), y_pred[i].tolist())
                        f.writelines( str(sku_id[i]) + '\t' + ",".join(s) + '\n')
                t2=time.time()
                print("write file cost %s sec" %(t2-t1))

        else:
            print('No checkpoint file found')
        coord.request_stop()
        coord.join(threads)


def main(argv=None):
    # 多进程：
    for idx,cid in enumerate(run_cid_list):
        # train(cid)
        print("start train cid=" + cid)
        p=Process(target=evaluate,args=(cid,))
        p.start()
        p.join()


if __name__ == '__main__':
    # tf.app.run()
    main()



