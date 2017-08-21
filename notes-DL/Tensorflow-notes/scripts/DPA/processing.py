# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
import cPickle as pickle

N=90
M=75  #多少个连续特征

MODEL_SAVE_PATH = "./models/"
PREPROCESSING_MODEL_PATH ="./models/preprocessing/"

# X_continous是所有的continus列
def standard_scaler(X_continous,flag):
    if flag =="fit":
        # 将连续型数据进行标准化
        x_mean = np.mean(X_continous, 0)
        x_std = np.std(X_continous, 0) + 1e-8
        with open(PREPROCESSING_MODEL_PATH + "ContinuousFeaturesStandard.pkl", "wb") as f:
            pickle.dump((x_mean, x_std), f)
    else:
        with open(PREPROCESSING_MODEL_PATH + "ContinuousFeaturesStandard.pkl", "rb") as f:
            x_mean, x_std = pickle.load(f)

    X_new = (X_continous - x_mean) / x_std
    return X_new

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

# def oneHot(X_category):
#     # 对离散型特征进行one-hot转化
#     with open(PREPROCESSING_MODEL_PATH + "CategoryFeaturesDict.pkl", "rb") as f:
#         vocabularies = pickle.load(f)
#
#     n_category_features = len(vocabularies)
#     X_category_onehot = []
#
#     #每一个分别进行onehot
#     for i in xrange(n_category_features):
#         onehot_i = tf.one_hot(tf.cast(X_category[:, i], tf.int64), depth=max(vocabularies[i].values())+2)
#         # x_category_ONE_HOT = tf.concat((x_category_ONE_HOT, onehot_i), 1)
#         X_category_onehot.append(onehot_i)
#
#     return  X_category_onehot


# 得到string index后的max值
def gen_index_max(X_category):
    x_max= np.max(X_category,0)
    x_max= x_max + 1
    with open(PREPROCESSING_MODEL_PATH + "StringIndexMax.pkl", "wb") as f:
        pickle.dump(x_max, f)


def oneHot(X_category):
    # 对离散型特征进行one-hot转化
    with open(PREPROCESSING_MODEL_PATH + "StringIndexMax.pkl", "rb") as f:
        x_max = pickle.load(f)

    n_category_features = len(x_max)
    X_category_onehot = []

    #每一个分别进行onehot
    for i in xrange(n_category_features):
        onehot_i = tf.one_hot(X_category[:, i], depth= x_max[i])
        # x_category_ONE_HOT = tf.concat((x_category_ONE_HOT, onehot_i), 1)
        X_category_onehot.append(onehot_i)

    return  X_category_onehot









