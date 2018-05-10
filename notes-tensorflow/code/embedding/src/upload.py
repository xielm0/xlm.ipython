# -*- coding:utf-8 -*-
import os
import time
from multiprocessing import Pool
import logging
from download import reset_dir,get_ip

# upload要先删除 hdfs上的文件，要避免重复删除。
# so,表是分区的，每个机器上传到独立的分区。
g_source_path= "../data/result"
g_lzo_path ="../data/result_lzo"
g_target_path="hdfs://ns3/user/jd_ad/ads_sz/app.db/app_szad_m_dyrec_sku2vec_embedding"
ip=get_ip()
ip=ip.split(".")[-1]
g_target_path=os.path.join(g_target_path,"host=%s"%ip)


def merge_file(dir_path,minsize=500000000):
    """
    合并小文件
    """
    logging.info("merge file:%s" %(dir_path))
    file_list = os.listdir(dir_path)
    file_list = filter(lambda x: x != "tmp", file_list)
    file_list = map(lambda x: os.path.join(dir_path , x), file_list)

    i=0
    tmp_path =os.path.join(dir_path,"tmp")
    if not os.path.exists(tmp_path):
        os.mkdir(tmp_path)
    else:
        os.system("/bin/rm %s/*" %(tmp_path))
    file_name=os.path.join(dir_path,"tmp","part-%s" %(  str(i).zfill(5)))
    size=0
    for file in file_list:
        size+=os.path.getsize(file)
        if size <minsize:
            os.system("cat %s >> %s" %(file, file_name))
        else:
            os.system("cat %s >> %s" %(file, file_name))
            i+=1
            size=0
            file_name=os.path.join(dir_path,"tmp","part-%s" %( str(i).zfill(5)))
    #删除之前的旧文件
    os.system("/bin/rm %s/*" %(dir_path))
    os.system("mv %s/* %s/" %(tmp_path,dir_path))
    os.system("/bin/rmdir %s" %tmp_path )

def lzo_file(file):
    #file是全路径
    # lzo
    file_name=file.split("/")[-1]
    lzo_file=os.path.join(g_lzo_path, file_name+".lzo")
    #
    cmd = 'lzop  %s -o %s' % (file,lzo_file)
    logging.info(cmd)
    os.system(cmd)
    return lzo_file


def lzo(parallel_nums):
    #
    reset_dir(g_lzo_path)
    # file_list
    file_list = os.listdir(g_source_path)
    file_list = map(lambda x: os.path.join(g_source_path , x), file_list)
    file_list.sort()
    #
    pool = Pool(parallel_nums)
    pool.map(func=lzo_file, iterable=file_list)

def upload_file(file):
    # upload
    cmd = 'hadoop fs -put %s  %s/' %(file, g_target_path)
    logging.info(cmd)
    os.system(cmd)

def upload(parallel_nums):
    # delete
    cmd ='hadoop fs -rm %s/*' % (g_target_path )
    logging.info(cmd)
    os.system( cmd )

    # file_list
    file_list = os.listdir(g_lzo_path)
    file_list = map(lambda x: os.path.join(g_lzo_path , x), file_list)
    file_list.sort()
    # parallel upload
    # 开启进程pool
    pool = Pool(parallel_nums)
    pool.map(func=upload_file, iterable=file_list)

def main():
    logging.basicConfig(level=logging.INFO)
    #
    t1 =time.time()
    lzo(30)
    upload(20)
    t2 = time.time()
    logging.info('upload cost %f sec' % (t2-t1))


if __name__ == '__main__':
    main()


