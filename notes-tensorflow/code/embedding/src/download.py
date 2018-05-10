# -*- coding:utf-8 -*-
from adsz_data.hdfsclient import get_client
from adsz_data.hdfsclient import list as ll
import os
import time
from multiprocessing import Pool
import socket
import logging
# import fcntl, struct

HDFS_USER = 'jd_ad'
HDFS_URLS = ['http://172.22.90.104:50070','http://172.22.90.103:50070']
worker_machine=["172.18.161.13","172.18.161.27","172.18.161.19","172.18.161.12"]
n_worker = len(worker_machine)

HDFS_TRAIN_PATH="ads_sz/app.db/app_szad_m_dyrec_sku2vec_train_tensor/train_data/"
HDFS_APPLY_PATH="ads_sz/app.db/app_szad_m_dyrec_sku2vec_train_tensor/apply_data/"
REMOTE_TRAIN_PATH="hdfs://ns3/user/jd_ad/"+HDFS_TRAIN_PATH
REMOTE_APPLY_PATH="hdfs://ns3/user/jd_ad/"+HDFS_APPLY_PATH

LOCAL_TRAIN_PATH="../data/train/"
LOCAL_APPLY_PATH= "../data/apply/"


def reset_dir(dir_path):
    if os.path.exists(dir_path):
        cmd="/bin/rm -r %s" %dir_path
        logging.info(cmd)
        os.system(cmd)
    os.mkdir(dir_path)

# CLASSPATH=$(${HADOOP_HDFS_HOME}/bin/hadoop classpath --glob) python download.py
def get_ip():
    """
    获取本机的ip
    :param ifname:
    :return: string : '172.18.161.27'
    """
    ifname='eth0'
    hostname = socket.gethostname()
    ip=socket.gethostbyname(hostname)

    # s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # ip=socket.inet_ntoa(fcntl.ioctl(s.fileno(), 0x8915, struct.pack('256s', ifname[:15]))[20:24])
    return ip


index=worker_machine.index(get_ip())

def getdirsize(dir):
    size = 0
    for root, dirs, files in os.walk(dir):
        size += sum([os.path.getsize(os.path.join(root, name)) for name in files])
    return size


def download_file(source_target):
    """
    下载一个文件,且完整路径
    如果是文件夹，也能一个文件夹全下载,但是并不是多进程，会非常慢
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
    file_list_hdfs = map(lambda name : os.path.join(hdfs_dir_path , name), file_list)
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

    #将hdfs上的数据下载本地
    # 开启多线程
    start_time = time.time()
    pool = Pool(30)
    logging.info('start downloading...')
    pool.map(func=download_file, iterable=download_file_list)
    pool.close()
    pool.join()
    duration = time.time() - start_time
    filesize = getdirsize(local_dir) / 1024 / 1024
    logging.info('download %dM files cost %fsec' % (filesize, duration))


def download_train():
    reset_dir(LOCAL_TRAIN_PATH)
    download_dir(HDFS_TRAIN_PATH,LOCAL_TRAIN_PATH)
def download_apply():
    reset_dir(LOCAL_APPLY_PATH)
    download_dir(HDFS_APPLY_PATH,LOCAL_APPLY_PATH,2)

def main():
    logging.basicConfig(level=logging.INFO)
    download_apply()


if __name__ == '__main__':
    main()