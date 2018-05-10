# -*- coding:utf-8 -*-
import tensorflow as tf
import pandas as pd
import download
import os
import time
import logging

column_nums=7

# TRAIN_DATA_PATH = download.LOCAL_TRAIN_PATH
TRAIN_DATA_PATH = download.REMOTE_TRAIN_PATH
APPLY_DATA_PATH = download.LOCAL_APPLY_PATH

# 读取hdfs原文件，LZO的文件已经解压
# CLASSPATH=$(${HADOOP_HDFS_HOME}/bin/hadoop classpath --glob) python input.py
def init_env():
    # 设置环境变量，让tensorflow能够访问hdfs
    cmd = os.environ['HADOOP_HDFS_HOME'] + '/bin/hadoop classpath --glob'
    CLASSPATH = os.popen(cmd).read()
    os.environ['CLASSPATH'] = CLASSPATH


def reset_dir(dir_path):
    if os.path.exists(dir_path):
        cmd="/bin/rm -r %s" %dir_path
        logging.info(cmd)
        os.system(cmd)
    os.mkdir(dir_path)

def get_train_batch( batch_size,dir_path= TRAIN_DATA_PATH):
    # files = ['../data/train/part-00000',]
    files = tf.train.match_filenames_once( os.path.join(dir_path , "part-*" ))
    filename_queue = tf.train.string_input_producer(files, shuffle=True, num_epochs=2)
    reader = tf.TextLineReader()
    key, value = reader.read_up_to(filename_queue, 10000)
    record_defaults = [[0] for _ in range(column_nums)]
    col = tf.decode_csv(records=value,
                        record_defaults=record_defaults,
                        field_delim='\t')
    # label = col[0]
    # feature =col[1:]
    label = tf.reshape(col[0],[-1,1])
    feature = tf.transpose(col[1:])

    label = tf.cast(label,tf.float32)
    feature = tf.cast(feature,tf.float32)
    feature_batch, label_batch = tf.train.shuffle_batch([feature, label],
                                                        batch_size=batch_size,
                                                        capacity=5000 + 3 * batch_size,
                                                        min_after_dequeue=5000,
                                                        num_threads=10,
                                                        enqueue_many=True,
                                                        allow_smaller_final_batch=False)

    return feature_batch, label_batch



def get_file_list(dir_path):
    file_list = os.listdir(dir_path)
    file_list = filter(lambda x: x[:4] == 'part', file_list)
    file_list = map(lambda x: os.path.join(dir_path , x), file_list)
    file_list.sort()
    return file_list


def pd_read_files(dir_path):
    t1 = time.time()
    file_list = os.listdir(dir_path)
    file_list = filter(lambda x: x[:4] == 'part', file_list)
    file_list = map(lambda x: dir_path + x, file_list)
    df_all=pd.DataFrame(columns=range(column_nums))
    for file in file_list:
        logging.info("reading file " + file)
        df = pd.read_csv(file, sep="\t", header=None, names=range(column_nums) )
        df_all = pd.concat([df_all, df])
    t2 = time.time()
    logging.info('read files cost %f sec' % (t2-t1))
    return df_all


def pd_read_data(dir_path):
    df = pd_read_files(dir_path )
    data = df.values
    data = data.astype(float)
    Y = data[:,0]
    X = data[:,1:]
    return X, Y


def get_apply_data(file):
    """
    读取的数据包括user_and_sku ,先取列，在转化为array ， 比先df.value，再 index要块很多。80w的数据能节省2s
    :param file:
    :return:
    """
    debug = False
    if debug:
        t1=time.time()
    logging.info("reading file " + file)
    df = pd.read_csv(file, sep="\t", header=None, names=range(column_nums) )

    if debug:
        t2=time.time()
        logging.info("read cost %s sec" %(t2-t1))
    # data = df.values
    sku_id = df.iloc[:,0].values
    # Y = data[:,2]
    X = df.iloc[:,1:].values
    X = X.astype(int)
    return sku_id, X

