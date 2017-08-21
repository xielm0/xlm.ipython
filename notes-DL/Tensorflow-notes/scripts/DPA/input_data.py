# -*- coding:utf-8 -*-
from datetime import datetime
import os
import time
import tensorflow as tf
import numpy as np
# import cPickle as pickle
# from sklearn.model_selection import train_test_split
import processing as pr


TRAIN_FILES_PATH = "hdfs://ns3/user/jd_ad/ads_sz/app.db/app_szad_m_dyrec_rank_train_tensorflow/type=train/part-*"
TEST_FILES_PATH =   "hdfs://ns3/user/jd_ad/ads_sz/app.db/app_szad_m_dyrec_rank_train_tensorflow/type=test/part-*"
#要求数据文件的格式是：label , continus_value, catagery_value
# 使用match_filenames_once，需要初始化变量。如果放在函数里，会出现初始化不成功。

#
N=90
M=75  #多少个连续特征


TF_TRAIN_0_PATH = "./data/DPA_data_0"
TF_TRAIN_1_PATH = "./data/DPA_data_1"
TF_TEST_PATH = "./data/DPA_data_test"



# 设置Tensorflow按需申请GPU资源
CONFIG = tf.ConfigProto(
    allow_soft_placement=True,
    log_device_placement=False)
CONFIG.gpu_options.allow_growth = True


# 读取hdfs原文件，LZO的文件已经解压
# CLASSPATH=$(${HADOOP_HDFS_HOME}/bin/hadoop classpath --glob) python input_data.py

def read_files_batch(files_path,num_records):
    files = tf.train.match_filenames_once(files_path)
    filename_queue = tf.train.string_input_producer(files , num_epochs=1 , shuffle=False )
    reader = tf.TextLineReader()
    key, value = reader.read(filename_queue)
    # 解析文件
    record_defaults = [["s"] for i in xrange(N)]
    col = tf.decode_csv(records=value,
                        record_defaults=record_defaults,
                        field_delim=',')
    label = tf.slice(col, begin=[0], size=[1])    # label为从第1列开始，共1列
    feature = tf.slice(col, begin=[1], size=[N-1])
    min_after_dequeue = 10000
    capacity = min_after_dequeue + 3 * num_records
    feature_bactch, label_batch = tf.train.batch([feature, label], batch_size=num_records, capacity=capacity, num_threads=16, allow_smaller_final_batch=True)
    return  feature_bactch,  label_batch


def read_files_all(files_path=TRAIN_FILES_PATH):
    num_records = 100000
    x, y = read_files_batch(files_path, num_records)
    X = []
    Y = []

    init_op = tf.global_variables_initializer()
    with tf.Session(config=CONFIG) as sess:
        # sess.run(init_op)
        sess.run(tf.local_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        while(1):
            print('loading data...')
            x_i, y_i = sess.run([x, y])
            print(x_i.shape)
            X += x_i.tolist()
            Y += y_i.tolist()
            if x_i.shape[0] < num_records:
                print('finished loading')
                break
        coord.request_stop()
        coord.join(threads)
    X = np.array(X)
    Y = np.array(Y)
    print(X.shape)
    return X, Y


"""
1.将数据预处理
2.将文件分割为train0 ， train1
2.将文件保存为TFrecords格式，便于tensorflow shuff batch
"""
def gen_train_tf():
    X, Y = read_files_all(TRAIN_FILES_PATH)

    # 数据类型
    Y=Y.astype(np.float32)
    X_continous = X[:, :M].astype(np.float32)
    X_category = X[:, M:].astype(np.float32)
    # 对数据进行归一化
    X_continous = pr.standard_scaler(X_continous,"fit")
    pr.gen_index_max(X_category)

    # 分解为train0和train1
    indice_0 = np.where(Y == 0)[0]
    indice_1 = np.where(Y == 1)[0]

    train0= np.concatenate( (Y[indice_0],X_continous[indice_0] ,X_category[indice_0] ),axis=1)
    train1= np.concatenate( (Y[indice_1],X_continous[indice_1] ,X_category[indice_1] ),axis=1)
    #保存为TFRecords
    write_TFRecords(train0,TF_TRAIN_0_PATH)
    write_TFRecords(train1,TF_TRAIN_1_PATH)


def gen_test_tf():
    X, Y = read_files_all(TEST_FILES_PATH)

    # 数据类型
    Y=Y.astype(np.float32)
    X_continous = X[:, :M].astype(np.float32)
    X_category = X[:, M:].astype(np.float32)
    # 对数据进行归一化
    X_continous = pr.standard_scaler(X_continous,"transform")

    test = np.concatenate( (Y,X_continous ,X_category ),axis=1)
    #保存为 TFRecords
    write_TFRecords(test,TF_TEST_PATH)




def _float64_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

# def _int64_feature(value):
#     return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

# def _bytes_feature(value):
#     return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))



def write_TFRecords(data,data_path):
    num_examples = data.shape[0]
    writer = tf.python_io.TFRecordWriter(data_path)
    for index in range(num_examples):
        #解析
        label= data[index,0]
        X_continous_s = data[index,1:M+1]
        X_category_s = data[index,M+1:]
        example = tf.train.Example(features=tf.train.Features(feature={
            'continous_feature': _float64_feature(X_continous_s),
            'category_feature': _float64_feature(X_category_s),
            'label': _float64_feature([label]) }))
        writer.write(example.SerializeToString())
    writer.close()

def read_TFRecords(files_path, batch_size):
    #files = tf.train.match_filenames_once( files_path)
    print([files_path])
    filename_queue = tf.train.string_input_producer([files_path])
    #
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
        serialized_example,
        features={
            'continous_feature': tf.FixedLenFeature([M], tf.float32),
            'category_feature': tf.FixedLenFeature([N-M-1], tf.float32),
            'label': tf.FixedLenFeature([1], tf.float32),
        })
    x_continues = features['continous_feature']
    x_category = tf.cast(features['category_feature'], tf.float32)
    feature = tf.concat([x_continues, x_category], 0)
    label = features['label']

    min_after_dequeue = 5000
    capacity = min_after_dequeue + 3 * batch_size
    feature_batch, label_batch = tf.train.shuffle_batch([feature, label], batch_size=batch_size,
                                                        capacity=capacity,
                                                        min_after_dequeue=5000,
                                                        num_threads=2,
                                                        allow_smaller_final_batch=True)

    return feature_batch, label_batch



def data_preprocessing():
    #
    gen_train_tf()
    gen_test_tf()


def get_train_data( batch_size):
    #采样比例
    label_rate = 0.2
    batch_size_single_label_0 = int(batch_size * (1 - label_rate))
    batch_size_single_label_1 = int(batch_size * label_rate)


    x0, y0 = read_TFRecords(TF_TRAIN_0_PATH, batch_size_single_label_0)
    y0 = tf.reshape(y0, (batch_size_single_label_0,))
    x1, y1 = read_TFRecords(TF_TRAIN_1_PATH, batch_size_single_label_1)
    y1 = tf.reshape(y1, (batch_size_single_label_1,))
    x = tf.concat([x0, x1], 0, name='x-input')
    y = tf.concat([y0, y1], 0, name='label-input')

    # 将x,y_的数据打乱
    indice = np.arange(batch_size)
    np.random.shuffle(indice)
    indices = tf.constant(indice, dtype=tf.int32)
    x_shuffle = tf.dynamic_stitch([indices], [x])
    y_shuffle = tf.dynamic_stitch([indices], [y])
    x = x_shuffle
    y = y_shuffle
    y = tf.cast(y, dtype=tf.float32)
    x_continuous = x[:, :M]
    x_category = x[:, M:]

    # 对离散型特征进行one-hot转化
    x_category_onehot=pr.oneHot(x_category)
    input_tensor = [x_continuous, x_category, x_category_onehot]

    return input_tensor,y

def get_eval_data(batch_size):
    x, y = read_TFRecords(TF_TEST_PATH, batch_size)
    return x,y


if __name__ == '__main__':
    data_preprocessing()





