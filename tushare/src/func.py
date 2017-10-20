# -*- coding:utf-8 -*-
import tushare as ts
import datetime
import numpy as np
import pandas as pd
import pickle
PREPROCESSING_MODEL_PATH ="../models/data_prepro_model/"

def get_trade_datelist():
    df = ts.get_k_data("hs300")
    date_list= df["date"].tolist()
    return  date_list

DATE_LIST = get_trade_datelist()

def get_trade_date(day,n):
    """
    计算钱n个交易日
    :param day:
    :param n:
    :return:
    """
    i=DATE_LIST.index(day)
    j = i+n
    return DATE_LIST[j]

now = datetime.datetime.now()
DAY = now.strftime('%Y-%m-%d')
DAY_01 = get_trade_date(DAY, -1)


def mkstring(x,sep):
    for i,xi in enumerate(x):
        if i ==0:
            s= str(xi)
        else:
            s = s + sep + str(xi)
    return s


def bucket(x, list1,n):
    """
    list1=[-0.75,-0.5,-0.25, 0, 0.25,0.5,0.75 ]
    len(list1)=7
    n=8
    bucket(0.4,list1,8)
    :return:
    """
    i=int(n/2)-1
    while 1:
        if i ==n-2 :
            return n-1
        elif i==0 :
            return  0
        elif x >=list1[i]:
            if x< list1[i+1]:
                return i+1
            else:
                i=i+1
        elif x < list1[i]:
            if x >= list1[i-1]:
                return i
            else :
                i=i-1


def df_to_vector(df):
    n= len(df)
    m= len(df.columns)
    x=[0 for i in range(n*m)]
    for i in range(n):
        for j in range(m):
            ind= i * m +j
            x[ind]=df.iloc[i,j]
    return x


def df_to_vector(df,n):
    """
    实际上df可能没有n行记录
    :param df:
    :param n:
    :return:
    """
    n2=len(df)
    m= len(df.columns)
    x=[0 for i in range(n*m)]
    n2 =min([n,n2])
    for i in range(n2):
        for j in range(m):
            ind= i * m +j
            x[ind]=df.iloc[i,j]
    return x

def series_to_vector(df,n):
    """
    实际上df可能没有n行记录
    :param df:
    :param n:
    :return:
    """
    x=[0 for i in range(n)]
    n2=len(df)
    n2 =min([n,n2])
    for i in range(n2):
            x[i]=df.iloc[i]
    return x


def string_index(X_category,flag):
    if flag =="fit":
        n_category_features = N-M-1
        vocabularies = []
        for ii in range(n_category_features):
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



#


