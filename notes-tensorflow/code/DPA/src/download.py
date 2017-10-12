# -*- coding:utf-8 -*-
from adsz_data.hdfsclient import get_client
from adsz_data.hdfsclient import list as ll
import os
import time
from multiprocessing import Pool
import socket, fcntl, struct

HDFS_USER = 'jd_ad'
HDFS_URLS = ['http://172.22.90.103:50070', 'http://172.22.90.104:50070']

worker_machine=["172.18.161.13","172.18.161.27","172.18.161.19","172.18.161.12"]

HDFS_TRAIN_PATH = '/user/jd_ad/ads_sz/app.db/app_szad_m_dyrec_rank_train_tensorflow/type=train_new/'
HDFS_TEST_PATH = '/user/jd_ad/ads_sz/app.db/app_szad_m_dyrec_rank_train_tensorflow/type=test_new/'
HDFS_APPLY_PATH = '/user/jd_ad/ads_sz/app.db/app_szad_m_dyrec_rank_train_tensorflow/type=apply_new/'

LOCAL_TRAIN_PATH = '../data/train/'
LOCAL_TEST_PATH = '../data/test/'
LOCAL_APPLY_PATH = '../data/apply/'

if not os.path.exists(LOCAL_TRAIN_PATH):
    os.mkdir(LOCAL_TRAIN_PATH)
if not os.path.exists(LOCAL_TEST_PATH):
    os.mkdir(LOCAL_TEST_PATH)
if not os.path.exists(LOCAL_APPLY_PATH):
    os.mkdir(LOCAL_APPLY_PATH)

# CLASSPATH=$(${HADOOP_HDFS_HOME}/bin/hadoop classpath --glob) python download.py

def get_ip():
    """
    获取本机的ip
    :param ifname:
    :return: string : '172.18.161.27'
    """
    ifname='eth0'
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    return socket.inet_ntoa(fcntl.ioctl(s.fileno(), 0x8915, struct.pack('256s', ifname[:15]))[20:24])

def getdirsize(dir):
    size = 0L
    for root, dirs, files in os.walk(dir):
        size += sum([os.path.getsize(os.path.join(root, name)) for name in files])
    return size


def download_file(source_target):
    """
    下载一个文件,且完整路径
    :param hdfs_local: hdfs_local=hdfs_file_path+"||"+local_dir
    :return:
    """
    hdfs_file=source_target.split("||")[0]
    local_dir=source_target.split("||")[1]
    #
    client = get_client(HDFS_URLS, HDFS_USER)
    file_name = hdfs_file.split('/')[-1]
    client.download(hdfs_file, os.path.join(local_dir, file_name))


def download_dir(hdfs_dir_path,local_dir,flag=1):
    """
    flag = 1 ,为单机下载； flag=2,为多机下载
    :param hdfs_dir_path:
    :param local_dir:
    :param flag:
    :return:
    """
    client = get_client(HDFS_URLS, HDFS_USER)
    file_list = ll(client, hdfs_dir_path)  # return  [part-0000, part-0001]
    del(client)
    #完整路径
    file_list_hdfs = map(lambda name : hdfs_dir_path + name, file_list)
    file_list_hdfs_local = map(lambda a : a +"||" + local_dir ,file_list_hdfs )
    #
    if flag ==1 :
        download_file_list=file_list_hdfs_local
    elif flag==2:
        # 获取当前机器的编号
        index=worker_machine.index(get_ip())
        n_worker = len(worker_machine)
        download_file_list=[]
        for i in range( len(file_list_hdfs_local)):
            if index == i % n_worker:
                download_file_list.append(file_list_hdfs_local[i])

    #
    print('delete path ' + local_dir)
    os.system('/bin/rm ' + local_dir + '*')
    #将hdfs上的数据下载本地
    # 开启多线程
    start_time = time.time()
    pool = Pool(36)
    print('start downloading...')
    pool.map(func=download_file, iterable=download_file_list)
    pool.close()
    pool.join()
    duration = time.time() - start_time
    filesize = getdirsize(local_dir) / 1024 / 1024
    print('download %dM files cost %fsec' % (filesize, duration))


def main():
    download_dir(HDFS_TRAIN_PATH,LOCAL_TRAIN_PATH)
    download_dir(HDFS_TEST_PATH,LOCAL_TEST_PATH)
    #download_dir(HDFS_APPLY_PATH,LOCAL_APPLY_PATH,flag=2)

if __name__ == '__main__':
    main()