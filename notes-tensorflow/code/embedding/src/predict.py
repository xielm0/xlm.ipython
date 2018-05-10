# -*- coding: utf-8 -*-
import time
import logging
import tensorflow as tf
import numpy as np
import train
import input
import model
import os
from multiprocessing import Process



# CLASSPATH=$(${HADOOP_HDFS_HOME}/bin/hadoop classpath --glob) python eval.py
LOCAL_RESULT_PATH = "../data/result"
N_GPU = 4
BATCH_SIZE = N_GPU*1000
discrete_nums=input.column_nums-1
PARALLEL=10

CONFIG = tf.ConfigProto(
    allow_soft_placement=True,
    log_device_placement=False)
CONFIG.gpu_options.allow_growth = True



def predict_with_one_gpu(x_, gpu_idx):
    """
    用1个gpu去predict
    """
    sku2vec=model.sku2vec(train.param_dict)
    with tf.device('/gpu:%d' % gpu_idx):
        # 前向传播
        y = sku2vec.inference(x_ )
    return y

def predict_with_multi_gpu(x_, batch_size, n_gpu):
    """
    用n个gpu去运行。这里只是构建了graph，这里的for循环只是构建了分支，在sess.run的时候是并行执行的。
    """
    sku2vec=model.sku2vec(train.param_dict)
    y_list = []
    batch_each= batch_size/n_gpu
    for i in range(n_gpu):
        with tf.device('/gpu:%d' % i):
            with tf.name_scope('GPU_%d' % i) as scope:
                c=i*batch_each
                d=(i+1)* batch_each if i < n_gpu-1 else batch_size
                x_i = x_[c:d, ]
                # 前向传播
                y_i = sku2vec.inference(x_i )
                tf.get_variable_scope()._reuse=True
                y_list.append(y_i)

    tf.get_variable_scope()._reuse=False
    y = tf.concat(y_list,0)

    return y

def evaluate(pid):
    """
    2种并行方案。
    1）4个gpu分别独立，每个gpu独立读取一个文件，独立计算一个文件的结果。
    2）将一个batch切分为1/4，每个gpu运行1/4。最后将结果合并。
    比较：
    第一种，4个gpu独立，则4个并行。加大batch_size，则gpu内存的利用率会更高。
    第二种，4个gpu共同执行一个任务，缺点：多了数据切分，合并等操作。
    方案1的并行度一般设置为N-gpu的倍数，如4，8，12；而方案2，可以设为6，10等。
    :param pid: 并行id 。
    :return:
    """
    file_list = input.get_file_list(input.APPLY_DATA_PATH)
    # build graph
    # 占位符
    x_ = tf.placeholder(tf.int32, shape=[None,discrete_nums], name='x-input')
    batch_ = tf.placeholder(tf.int32, [ ], name='actual_bach_size')

    # 前向传播
    y = predict_with_multi_gpu(x_,BATCH_SIZE,N_GPU)

    #
    saver = tf.train.Saver()

    with tf.Session(config=CONFIG) as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        # tf.train.get_checkpoint_state 会自动找到目录中最新模型的文件名。
        model_save_path =os.path.join(train.MODEL_SAVE_PATH)
        ckpt = tf.train.get_checkpoint_state(model_save_path)
        if ckpt and ckpt.model_checkpoint_path:
            # 加栽变量
            saver.restore(sess, ckpt.model_checkpoint_path)

            for idx,file in enumerate(file_list):
                if idx % PARALLEL != pid :
                    continue
                #
                sku_id, X = input.get_apply_data(file)

                # 计算epoch_num
                n = len(sku_id)
                epoch_num = n / BATCH_SIZE if n % BATCH_SIZE == 0 else (n / BATCH_SIZE) + 1
                logging.info("n= %s , epoch_num= %s " %(n, epoch_num))

                t1 = time.time()
                y_pred = []
                for i in range(epoch_num):
                    a = i* BATCH_SIZE
                    b = (i+1)* BATCH_SIZE
                    b = b if b<=n else n
                    x_batch = X[a:b,]
                    batch_size = b-a
                    y_p = sess.run(y,feed_dict={x_: x_batch, batch_:batch_size})
                    y_pred += y_p.tolist()
                y_pred = np.array(y_pred)
                logging.info(y_pred.shape)
                t2 = time.time()
                logging.info("eval cost %s sec" %(t2-t1))

                # save
                file_name = file.split("/")[-1]
                result_file = os.path.join(LOCAL_RESULT_PATH,file_name)
                #
                t1=time.time()
                write_result(sku_id,y_pred,result_file)
                t2=time.time()
                logging.info("write file cost %s sec" %(t2-t1))

        else:
            logging.info('No checkpoint file found')
        coord.request_stop()
        coord.join(threads)

def write_result(id,vec,file_name):
    np.set_printoptions(precision=8)
    #
    with open(file_name, 'w') as f:
        for i in xrange(len(id) ):
            s=map(lambda x: str(x), vec[i].tolist())
            f.writelines( str(id[i]) + '\t' + ",".join(s) + '\n')



def main(argv=None):
    logging.basicConfig(level=logging.INFO)
    #
    input.reset_dir(LOCAL_RESULT_PATH)

    # evaluate(0)
    process_list = []
    for i in range(PARALLEL):
        p=Process(target=evaluate,args=(i,))
        p.start()
        process_list.append(p)
    for p in process_list:
        p.join()


if __name__ == '__main__':
    # tf.app.run()
    main()



