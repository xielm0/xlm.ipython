# -*- coding:utf-8 -*-
import os
import time
import predict
from multiprocessing import Pool
from log import logger

cid_list=predict.cid_list
run_cid_list=predict.run_cid_list
LOCAL_RESULT_PATH=predict.LOCAL_RESULT_PATH
LOCAL_LZO_PATH="../data/result_lzo"
TARGET_PATH="hdfs://ns3/user/jd_ad/ads_sz/app.db/app_szad_m_dyrec_sku2vec_embedding"


if not os.path.exists(LOCAL_LZO_PATH):
    os.mkdir(LOCAL_LZO_PATH)

for cid in run_cid_list:
    path = os.path.join(LOCAL_LZO_PATH,cid)
    if not os.path.exists(path):
        os.mkdir(path)


def upload(file):
    """
    file是全路径
    :param file:
    :return:
    """
    t1 = time.time()
    # lzo
    command1 = 'lzop  %s' % file
    logger.info(command1)
    os.system(command1)
    #
    file_name=file.split("/")[-1]
    cid =file.split("/")[-2]
    lzo_path=os.path.join(LOCAL_LZO_PATH,cid)
    lzo_file=os.path.join(lzo_path,cid+"-"+file_name+".lzo")
    command2 = 'mv %s.lzo %s' % (file, lzo_file)
    logger.info(command2)
    os.system(command2)

    # upload
    command4 = 'hadoop fs -put %s  %s/' %(lzo_file, TARGET_PATH)
    logger.info(command4)
    os.system(command4)

    t2 = time.time()
    logger.info('upload costs %fsec' % (t2-t1))


#合并小文件
def merge_file(dir_path):
    logger.info("merge file:%s" %(dir_path))
    file_list = os.listdir(dir_path)
    file_list = filter(lambda x: x != "tmp", file_list)
    file_list = map(lambda x: os.path.join(dir_path , x), file_list)
    # cid=dir_path.split("/")[-1]

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
        if size <500000000:
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


def main():
    # delete
    logger.info("delete exists files, including hdfs files")
    for cid in run_cid_list:
        os.system('/bin/rm  %s/*.lzo' % ( os.path.join(LOCAL_LZO_PATH , cid)))
    # 将hdfs上分区的数据先删除
    os.system( 'hadoop fs -rm %s/%s-*' % (TARGET_PATH ,cid) )

    #merge
    for cid in run_cid_list:
        dir_path=os.path.join(LOCAL_RESULT_PATH,cid)
        merge_file(dir_path)

    # file_list
    all_file_list=[]
    for cid in run_cid_list:
        dir_path=os.path.join(LOCAL_RESULT_PATH,cid)
        file_list = os.listdir(dir_path)
        file_list = map(lambda x: os.path.join(dir_path , x), file_list)
        all_file_list += file_list

    # parallel upload
    # 开启进程pool
    pool = Pool(8)
    logger.info('start upload...')
    pool.map(func=upload, iterable=all_file_list)



if __name__ == '__main__':
    main()


