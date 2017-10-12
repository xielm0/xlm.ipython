# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
# import cPickle as pickle
# from sklearn.model_selection import train_test_split
import processing as pr
import pandas as pd
import download
import os
import time
import logging

TRAIN_FILES_PATH = "hdfs://ns3/user/jd_ad/ads_sz/app.db/app_szad_m_dyrec_rank_train_tensorflow/type=train_new/part-*"
TEST_FILES_PATH = "hdfs://ns3/user/jd_ad/ads_sz/app.db/app_szad_m_dyrec_rank_train_tensorflow/type=test_new/part-*"
#要求数据文件的格式是：label , continus_value, catagery_value
# 使用match_filenames_once，需要初始化变量。如果放在函数里，会出现初始化不成功。

LOCAL_TRAIN_PATH = download.LOCAL_TRAIN_PATH
LOCAL_TEST_PATH = download.LOCAL_TEST_PATH
LOCAL_APPLY_PATH = download.LOCAL_APPLY_PATH

#
N=pr.N  #多少列，第1个为label
M=pr.M  #多少个连续特征


TF_TRAIN_0_PATH = "../data/DPA_data_0"
TF_TRAIN_1_PATH = "../data/DPA_data_1"
TF_TEST_PATH = "../data/DPA_data_test"


# 设置Tensorflow按需申请GPU资源
CONFIG = tf.ConfigProto(
    allow_soft_placement=True,
    log_device_placement=False)
CONFIG.gpu_options.allow_growth = True


# 读取hdfs原文件，LZO的文件已经解压
# CLASSPATH=$(${HADOOP_HDFS_HOME}/bin/hadoop classpath --glob) python input.py

"""
tensflow提供了reader,他支持的shuffle_batch非常适合 train的时候。
但在apply的时候，reader的读取性能非常差，尝试并行使用多个reader，还是效果不好。
so,在apply的时候采用pandas读取。

思路一：train:
pd.readfile=> array[string] =>
data preprocessing ： onehot需要max value ,需要读取所有数据，so,这里需要readfile => array ,全部数据。也可以分布式算法计算，或者抽样
split:y=0&y=1=>covert2TFRecoard=>writeTFRecoard
read_TFRrecord : batch , shuffle batch  : return tensor
train

思路2：apply的时候：
readline
每次读10000行， 读epoch次  : return ndarray
data preprocessing
inference

"""


def read_csv(files_path,batch_size):
    files = tf.train.match_filenames_once(files_path)
    filename_queue = tf.train.string_input_producer(files , num_epochs=1 , shuffle=False )
    reader = tf.TextLineReader()
    key, value = reader.read(filename_queue)
    # 解析文件
    record_defaults = [["s"] for i in range(N)]
    data = tf.decode_csv(records=value,
                         record_defaults=record_defaults,
                         field_delim=',')
    min_after_dequeue = 10000
    capacity = min_after_dequeue + 3 * batch_size
    batch_data = tf.train.batch([data], batch_size=batch_size, capacity=capacity, num_threads=4, allow_smaller_final_batch=True)
    return  batch_data


def read_TFRecords(files_path, batch_size ):
    """
    1个reader使用1个cpu资源，一个cpu只支持4个线程。
    return tensor
    """
    # files = tf.train.match_filenames_once( files_path)
    # print([files_path])
    filename_queue = tf.train.string_input_producer([files_path])
    #
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
        serialized_example,
        features={
            'label': tf.FixedLenFeature([1], tf.float32),
            'feature': tf.FixedLenFeature([N-1], tf.float32),
        })
    # features = [tf.parse_single_example(
    #     serialized_example,
    #     features={
    #         'label': tf.FixedLenFeature([1], tf.float32),
    #         'feature': tf.FixedLenFeature([N-1], tf.float32),
    #     }) for _ in range(n_reader)]

    min_after_dequeue = 5000
    capacity = min_after_dequeue + 3 * batch_size
    features_batch = tf.train.shuffle_batch(features, batch_size=batch_size,
                                                        capacity=capacity,
                                                        min_after_dequeue=5000,
                                                        num_threads=4,
                                                        allow_smaller_final_batch=True)

    feature_batch = features_batch['feature']
    label_batch = features_batch['label']

    return feature_batch, label_batch


def get_file_list(dir_path):
    file_list = os.listdir(dir_path)
    file_list = filter(lambda x: x[:4] == 'part', file_list)
    file_list = map(lambda x: dir_path + x, file_list)
    return file_list


def pd_read_files(dir_path):
    t1 = time.time()
    file_list = os.listdir(dir_path)
    file_list = filter(lambda x: x[:4] == 'part', file_list)
    file_list = map(lambda x: dir_path + x, file_list)
    df_all=pd.DataFrame(columns=range(N))
    for file in file_list:
        print("reading file " + file)
        df = pd.read_csv(file, sep=",", header=None, names=range(N) )
        df_all = pd.concat([df_all, df])
    t2 = time.time()
    print('read files cost %f sec' % (t2-t1))
    return df_all


# 下载到本地，在通过pd.read_csv()读取，这个速度快
def maybe_download():
    download.download_dir(download.HDFS_TRAIN_PATH)


# def _int64_feature(value):
#     return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float64_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


# 将txt转换成tfrecord格式
def covert2tfrecord(txt_file,tf_record):
    writer = tf.python_io.TFRecordWriter(tf_record)
    with open(txt_file) as f:
        lines=f.readlines()
        for line in lines:
            item=line.split(",").astype(float)
            label = item[0]
            feature = item[1:]
            example = tf.train.Example(features=tf.train.Features(feature={
                'label': _float64_feature(label),
                'feature': _float64_feature(feature)
            }))
            writer.write(example.SerializeToString())

    writer.close()


# 要求 data 是 array ,且元素是： float
def write_TFRecords(data,TFRecoard_path):
    writer = tf.python_io.TFRecordWriter(TFRecoard_path)
    for data_i in data:
        #解析
        label = data_i[0]
        feature = data_i[1:]
        example = tf.train.Example(features=tf.train.Features(feature={
            'label': _float64_feature([label]),
            'feature': _float64_feature(feature)
             }))
        writer.write(example.SerializeToString())
    writer.close()


def gen_train_tf():
    t1=time.time()
    # 通过pd快读读取本地文件,读取的是文本,但都是数字
    df = pd_read_files(LOCAL_TRAIN_PATH )
    data = df.values
    # data = read_files_all(TRAIN_FILES_PATH)
    data = data.astype(float)

    t2=time.time()
    print('read cost %fsec' % (t2-t1))
    # 数据预处理
    Y = data[:, 0]
    X = data[:, 1:]
    pr.input_fn(X, "fit")

    t3=time.time()
    print('data preprocessing cost %f sec' % (t3-t2))

    #分解为train0和train1
    indice_0 = np.where(Y == 0)[0]
    indice_1 = np.where(Y == 1)[0]
    #
    train0=data[indice_0, ]
    train1=data[indice_1, ]

    t4=time.time()
    print('split cost %f sec' % (t4-t3))
    # 保存为TFRecords
    write_TFRecords(train0, TF_TRAIN_0_PATH)
    write_TFRecords(train1, TF_TRAIN_1_PATH)

    t5=time.time()
    print('write_TFRecords cost %f sec' % (t5-t4))


def gen_test_tf():
    t1 = time.time()
    df = pd_read_files(LOCAL_TEST_PATH)
    data = df.values
    # data = read_files_all(TEST_FILES_PATH)
    test = data.astype(float)
    # 保存为 TFRecords
    write_TFRecords(test, TF_TEST_PATH)

    t5=time.time()
    print('write_TFRecords cost %f sec' % (t5-t1))


def get_train_data(batch_size):
    """
    按照比例采样，并打乱排序
    :param batch_size:
    :return:
    """
    #采样比例
    label_rate = 0.2
    batch_size_single_label_0 = int(batch_size * (1 - label_rate))
    batch_size_single_label_1 = int(batch_size * label_rate)

    x0, y0 = read_TFRecords(TF_TRAIN_0_PATH, batch_size_single_label_0)
    x1, y1 = read_TFRecords(TF_TRAIN_1_PATH, batch_size_single_label_1)
    x = tf.concat([x0, x1], 0, name='x-input')
    y = tf.concat([y0, y1], 0, name='y-input')

    # 打乱排序
    indice = np.arange(batch_size)
    np.random.shuffle(indice)
    indices = tf.constant(indice, dtype=tf.int32)
    x_shuffle = tf.dynamic_stitch([indices], [x]) # x_shuffle =x[indices]
    y_shuffle = tf.dynamic_stitch([indices], [y])
    x = x_shuffle # tensor
    y = y_shuffle
    # x_category => embedding
    x_list =pr.input_fn(x,"apply")

    return x_list, y


def get_test_data():
    """
    :return: array
    """
    df = pd_read_files(LOCAL_TEST_PATH )
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
    print("reading file " + file)
    df = pd.read_csv(file, sep=",", header=None, names=range(N+2) )

    if debug:
        t2=time.time()
        print("read cost %s sec" %(t2-t1))
    # data = df.values
    user_and_sku = df.iloc[:,0:2].values
    # Y = data[:,2]
    X = df.iloc[:,3:].values
    X = X.astype(float)
    return user_and_sku, X


def main():
    gen_train_tf()
    gen_test_tf()

if __name__ == '__main__':
    main()
