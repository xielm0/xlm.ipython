# -*- coding: utf-8 -*-
import time
import tensorflow as tf
import numpy as np
import train
import input
import pandas as pd
import os
from multiprocessing import Process

"""
1.sku的文件按三级类目进行partition, 这里的细节是：每个三级类目是一个文件
2.读取一个文件的embedding,生成一个矩阵A
3, 对每个sku,计算最相似的topk. 根据公式：cosine=b*A.t
"""
# CLASSPATH=$(${HADOOP_HDFS_HOME}/bin/hadoop classpath --glob) python eval.py
LOCAL_APPLY_PATH =input.LOCAL_APPLY_PATH
LOCAL_RESULT_PATH = "../data/result/"
LOCAL_SYNONYMS_PATH="../data/synonyms/"
N_GPU = 4
PARALLEL=4

CONFIG = tf.ConfigProto(
    allow_soft_placement=True,
    log_device_placement=False)
CONFIG.gpu_options.allow_growth = True

if not os.path.exists(LOCAL_SYNONYMS_PATH):
    os.mkdir(LOCAL_SYNONYMS_PATH)

discrete_nums=train.discrete_nums
vocabulary_size=train.vocabulary_size
embedding_size=train.embedding_size
embedding_size_list=train.embedding_size_list


def get_embedding(file_path):
    #
    with open(file_path,"r") as f:
        lines=f.readlines()
    n = len(lines)
    sku_list = []
    embeddings =[]
    for line in lines:
        s=line.split("\t")
        sku_list.append(s[0])
        sku_embedding =  map(lambda x: float(x) ,s[1].split(","))
        #由于是同一个三级类目下，so,只需要后面的一部分
        sku_embedding = sku_embedding[36:]
        embeddings.append(sku_embedding)
    embeddings =np.array(embeddings)
    return sku_list, embeddings

def norm_embedding(embeddings):
    # norm
    norm = np.sqrt((embeddings ** 2).sum(axis =1 , keepdims=True))
    normalized_embeddings = embeddings / norm

    return normalized_embeddings


def findSynonyms(sku_list,embeddings,file_name,top_k=10):
    n=len(sku_list)
    result_file = os.path.join(LOCAL_SYNONYMS_PATH, file_name)
    t0 =time.time()

    f= open(result_file, 'w')
    for i in range(n):
        sku_embedding=embeddings[i]
        #cosine
        sim= np.dot(embeddings,sku_embedding.T)  # retrun shape = [vocabsize,1]
        nearest = (-sim).argsort()[1:top_k+1]

        str=[]
        for k in range(top_k):
            str.append(sku_list[nearest[k]] )
        s=sku_list[i] + "\t" + ",".join(str)
        # log
        if 0:
            print(s)
        # save
        f.writelines(s+"\n")
    f.close()
    print("write file: %s, cost %s sec " %(result_file,time.time()-t0))

from multiprocessing.pool import ThreadPool

def findSynonyms_batch(sku_list,embeddings,file_name,top_k=10):
    n=len(sku_list)
    result_file = os.path.join(LOCAL_SYNONYMS_PATH, file_name)
    f= open(result_file, 'w')
    t0=time.time()
    # 按batch进行扫描
    batch_size =1000
    epoch_num = n / batch_size if n % batch_size == 0 else (n / batch_size) + 1
    for i in range(epoch_num):
        t1 =time.time()
        a = i* batch_size
        b = (i+1)* batch_size
        b = b if b<=n else n

        sku_embedding=embeddings[a:b,]
        #cosine
        sim= np.dot(sku_embedding,embeddings.T)  # retrun shape = [batch,vocabsize]


        def topk(j):
            nearest = (-sim[j,:]).argsort()[1:top_k+1]  # nearst是主要的耗时环节，占99%
            # str = sku_list[nearest].tolist()
            str=[]
            for k in range(top_k):
                str.append(sku_list[nearest[k]] )
            s=sku_list[a+j] + "\t" + ",".join(str)
            return s

        pool.map

        for j in range(b-a):
            nearest = (-sim[j,:]).argsort()[1:top_k+1]  # nearst是主要的耗时环节，占99%
            # nearest = np.argsort(-sim[j,:]，axis=-1)[1:top_k+1]

            # 如果sku_list是一个array，可以直接得到topK的sku
            # str = sku_list[nearest].tolist()
            str=[]
            for k in range(top_k):
                str.append(sku_list[nearest[k]] )
            s=sku_list[a+j] + "\t" + ",".join(str)
            # log
            if 0:
                print(s)
            # save
            f.writelines(s+"\n")

        # print("batch cost %s sec:" %(time.time()-t1))

    f.close()
    t1=time.time()
    print("write file: %s, cost %s sec " %(result_file,t1-t0))


def parallel_findSynonyms(pid):
    #
    file_list = input.get_file_list(LOCAL_RESULT_PATH)
    #
    for idx,file in enumerate(file_list):
        if idx % PARALLEL != pid :
            continue
        #read
        t0 = time.time()
        sku_list, embeddings=get_embedding(file)
        t1 = time.time()
        print("read file: %s ,cost %s sec" %(file,(t1-t0)))
        #
        normalized_embeddings=norm_embedding(embeddings)
        #
        file_name = file.split("/")[-1]
        findSynonyms_batch(sku_list, normalized_embeddings,file_name, 10)


def main(argv=None):
    pid_list=range(16)
    for i in pid_list:
        p=Process(target=parallel_findSynonyms,args=(i,))
        p.start()


if __name__ == '__main__':
    # tf.app.run()
    main()



