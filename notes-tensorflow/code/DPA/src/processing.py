# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
import cPickle as pickle

N=44
M=18  #多少个连续特征

PREPROCESSING_MODEL_PATH ="../models/data_prepro_model/"


def standard_scaler_fit(X_continue):
    """
    fit的时候，X_continue是所有的数据，而apply的时候，是一个batch数据。
    :param X_continue: X_continue是所有的continue列
    :return:
    """
    # 将连续型数据进行标准化
    x_mean = np.mean(X_continue, 0)
    x_std = np.std(X_continue, 0) + 1e-8
    with open(PREPROCESSING_MODEL_PATH + "ContinuousFeaturesStandard.pkl", "wb") as f:
        pickle.dump((x_mean, x_std), f)
    X_new = X_continue
    return X_new


def standard_scaler_transform(X_continue, x_mean, x_std):
    X_new = (X_continue - x_mean) / x_std
    return X_new

def standard_scaler_get():
    """
    apply的时候，是一个batch数据,不适合频繁的open file.只需要open一次，然后传入到standard_scaler_transform就好。
    :return:
    """
    with open(PREPROCESSING_MODEL_PATH + "ContinuousFeaturesStandard.pkl", "rb") as f:
        x_mean, x_std = pickle.load(f)
    return x_mean, x_std



def string_index(X_category,flag):
    if flag =="fit":
        n_category_features = N-M-1
        vocabularies = []
        for ii in xrange(n_category_features):
            feat_value = X_category[:, ii]
            feat_value = set(feat_value)
            feat_value_index = range(len(feat_value))
            vocab = dict(zip(feat_value, feat_value_index))
            vocabularies.append(vocab)

        with open(PREPROCESSING_MODEL_PATH + "CategoryFeaturesDict.pkl", "wb") as f:
            pickle.dump(vocabularies, f)
    else:
        with open(PREPROCESSING_MODEL_PATH + "CategoryFeaturesDict.pkl", "rb") as f:
            vocabularies = pickle.load(f)

    for ii in xrange(n_category_features):
        vocab = vocabularies[ii]
        X_category[:, ii] = np.array(map(lambda xx: vocab[xx] if vocab.has_key(xx) else max(vocab.values())+1,
                                         X_category[:, ii]))
    X_new = X_category.astype(np.int64)
    return X_new


def index_max_fit(X_category):
    x_max= np.max(X_category,0)
    x_max= x_max + 1
    with open(PREPROCESSING_MODEL_PATH + "StringIndexMax.pkl", "wb") as f:
        pickle.dump(x_max, f)
    return x_max


def index_max_get():
    with open(PREPROCESSING_MODEL_PATH + "StringIndexMax.pkl", "rb") as f:
        x_max = pickle.load(f)
    return x_max



def oneHot(X_category, x_max):
    """
    对离散型特征进行one-hot转化
    :param X_category:
    :param x_max:
    :return:
    """
    n_category_features = len(x_max)
    X_category_onehot = []
    # 每一个category_feature 分别进行onehot
    for i in xrange(n_category_features):
        onehot_i = tf.one_hot(tf.cast(X_category[:, i],tf.int32), depth=int(x_max[i]))
        X_category_onehot.append(onehot_i)

    return  X_category_onehot

def cross_str(a,b,hash_bucket_size):
    c=a+":"+b
    return hash(c) % hash_bucket_size

def cross_column(x1,x2,num_buckets):
    """
    hash("d"+":"+"a") % hash_bucket_size
    :param x1: 离散特征1 ，输入为tensor ,x1=np.array([1.,2.,3.]) or x1 = tf.constant(x1)
    :param x2: 离散特征2 ，输入为tensor ,x2=np.array([2.,3.,4.]) or x2 = tf.constant(x2)
    :return: cross feature
    """
    a= tf.as_string(tf.cast(x1,tf.int32))
    b= tf.as_string(tf.cast(x2,tf.int32))
    # reduce_join, string_join
    c= tf.string_join([a,b],separator=":")
    return tf.string_to_hash_bucket(c,num_buckets=num_buckets)


def get_processing():
    x_mean, x_std = standard_scaler_get()
    x_max = index_max_get()
    return x_mean,x_std,x_max

X_MEAN, X_STD,X_MAX = get_processing()


def input_fn(x_feature,flag="fit"):
    """
    1. 对连续数据进行标准化，获取mean ,和std
    2. 对分类数据进行index_max ,保存 index_max
    :param x_feature: when flag="fit", type is ndarray , when flag="apply" ，type is tensor
    :return: Null
    """
    x_continue = x_feature[:, :M]
    x_category = x_feature[:, M:]

    if flag =="fit" :
        standard_scaler_fit(x_continue)
        index_max_fit(x_category)
    else :
        x_continue=standard_scaler_transform(x_continue,X_MEAN,X_STD )
        x_max=X_MAX
        x_list=[x_continue,x_category,x_max]
        return x_list





