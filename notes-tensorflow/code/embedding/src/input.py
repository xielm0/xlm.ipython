# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
# import cPickle as pickle
# from sklearn.model_selection import train_test_split
import pandas as pd
import download
import os
import time
import logging
from multiprocessing import Pool


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

N=6
LOCAL_TRAIN_PATH = download.LOCAL_TRAIN_PATH
LOCAL_APPLY_PATH = download.LOCAL_APPLY_PATH
cid_list=download.cid_list

TF_TRAIN_PATH ="../data/TF/"

if not os.path.exists(TF_TRAIN_PATH):
    os.mkdir(TF_TRAIN_PATH)

for cid in cid_list:
    path = os.path.join(TF_TRAIN_PATH,cid)
    if not os.path.exists(path):
        os.mkdir(path)




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
        df = pd.read_csv(file, sep="\t", header=None, names=range(N) )
        df_all = pd.concat([df_all, df])
    t2 = time.time()
    print('read files cost %f sec' % (t2-t1))
    return df_all


def get_data(dir_path):
    df = pd_read_files(dir_path )
    data = df.values
    data = data.astype(float)
    Y = data[:,0]
    X = data[:,1:]
    return X, Y

# _index_in_epoch=0
# _num_examples=0
# _epochs_complete=0
# def next_batch(batch_size):
#     start = _index_in_epoch
#     _index_in_epoch += batch_size
#     if _index_in_epoch > _num_examples: # epoch中的句子下标是否大于所有语料的个数，如果为True,开始新一轮的遍历
#         # Finished epoch
#         _epochs_completed += 1
#
#         # Start next epoch
#         start = 0
#         _index_in_epoch = batch_size
#         assert batch_size <= _num_examples
#     end = _index_in_epoch
#     return data[start:end]



def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


# 要求 data 是 array ,且元素是： float
def write_TFRecords(data,TFRecoard_path):
    writer = tf.python_io.TFRecordWriter(TFRecoard_path)
    for data_i in data:
        #解析
        label = data_i[0]
        feature = data_i[1:]
        example = tf.train.Example(features=tf.train.Features(feature={
            'label': _float_feature([label]),
            'feature': _float_feature(feature)
        }))
        writer.write(example.SerializeToString())
    writer.close()


def gen_a_tf(arg_tuple):
        txt_file, tensor_file = arg_tuple[:2]
        print("reading file " + txt_file)
        df = pd.read_csv(txt_file, sep="\t", header=None, names=range(N) )
        data = df.values
        # 保存为 TFRecords
        write_TFRecords(data, tensor_file)

def gen_dir_tf(txt_dir, tensor_dir):
    file_list = os.listdir(txt_dir)
    file_list = filter(lambda x: x[:4] == 'part', file_list)
    txt_list = map( lambda file:os.path.join(txt_dir,file),file_list)
    tesor_list= map( lambda file: os.path.join(tensor_dir,file),file_list)
    zip_list = zip(txt_list,tesor_list)
    # print(zip_list)
    pool = Pool(32)
    pool.map(gen_a_tf,zip_list)


def gen_train_tf():
    txt_dir_list=download.get_recur_dir(LOCAL_TRAIN_PATH)
    tf_dir_list=download.get_recur_dir(TF_TRAIN_PATH)
    for i in range(len(txt_dir_list)):
        print(txt_dir_list[i],tf_dir_list[i])
        gen_dir_tf(txt_dir_list[i],tf_dir_list[i])


def read_format(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    # 解析文件
    features = tf.parse_single_example(
        serialized_example,
        features={
            'label': tf.FixedLenFeature([1], tf.float32),
            'feature': tf.FixedLenFeature([N-1], tf.float32),
        })
    return features


def read_TFRecords( batch_size ,cid):
    """
    1个reader使用1个cpu资源，一个cpu只支持4个线程。
    return tensor
    """
    dir_path=TF_TRAIN_PATH+"/%s"%cid +"/"
    files = tf.train.match_filenames_once( dir_path + "part-*")
    # files = ['../data/TF/part-00000',]
    filename_queue = tf.train.string_input_producer(files)
    #
    features = read_format(filename_queue)

    min_after_dequeue = 5000
    capacity = min_after_dequeue + 3 * batch_size
    features_batch = tf.train.shuffle_batch(features,
                                            batch_size=batch_size,
                                            capacity=capacity,
                                            min_after_dequeue=min_after_dequeue,
                                            num_threads=4,
                                            allow_smaller_final_batch=False)

    feature_batch = tf.cast(features_batch['feature'],tf.int32)
    label_batch =  tf.cast(features_batch['label'],tf.int32)

    return feature_batch, label_batch

def get_max(cid):
    index_max_file =os.path.join(download.LOCAL_DICT_PATH ,cid,"part-00000")
    df = pd.read_csv(index_max_file, sep="\t", header=None, names=range(5) )
    max=df.values
    return max[0]


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
    df = pd.read_csv(file, sep="\t", header=None, names=range(N) )

    if debug:
        t2=time.time()
        print("read cost %s sec" %(t2-t1))
    # data = df.values
    sku_id = df.iloc[:,0].values
    # Y = data[:,2]
    X = df.iloc[:,1:].values
    X = X.astype(int)
    return sku_id, X


def main():
    gen_train_tf()

if __name__ == '__main__':
    main()
