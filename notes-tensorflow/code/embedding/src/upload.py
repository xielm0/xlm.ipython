# -*- coding:utf-8 -*-
import os
import time
import apply
import download
import input
from multiprocessing import Process

RESULT_PATH=apply.LOCAL_RESULT_PATH
LZO_PATH=os.path.join(RESULT_PATH,"lzo/")
if not os.path.exists(LZO_PATH):
    os.mkdir(LZO_PATH)

PARALLEL=16

flag=download.get_ip()
TARGET_PATH='hdfs://ns3/user/jd_ad/ads_sz/app.db/app_szad_m_dyrec_rank_predict_res_tf/flag=%s' % flag

def lzo():
    """

    :return:
    """
    result_path=apply.LOCAL_RESULT_PATH
    lzo_path=os.path.join(result_path,"lzo")
    if not os.path.exists(lzo_path):
        os.mkdir(lzo_path)

    # delete
    os.system("/bin/rm %s/*.lzo" %(result_path) )
    os.system("/bin/rm %s/*.lzo" %(lzo_path) )



def upload(file_list, pid):
    t1 = time.time()
    for idx,file in enumerate(file_list):
        if idx % PARALLEL != pid :
            continue

        file_name=file.split("/")[-1]
        # lzo
        command1 = 'lzop  %s' % file
        print(command1)
        os.system(command1)
        #
        lzo_file=LZO_PATH +file_name+".lzo"
        command2 = 'mv %s.lzo %s' % (file, LZO_PATH)
        print(command2)
        os.system(command2)

        # upload
        command4 = 'hadoop fs -put %s  %s/' %(lzo_file, TARGET_PATH)
        print(command4)
        os.system(command4)

    t2 = time.time()
    print('upload costs %fsec' % (t2-t1))


def main():
    # delete
    print("delete exists files, including hdfs files")
    os.system("/bin/rm %s/*.lzo" %(RESULT_PATH) )
    os.system("/bin/rm %s/*.lzo" %(LZO_PATH) )
    # 将hdfs上分区的数据先删除
    os.system( 'hadoop fs -rm %s/*' % (TARGET_PATH ) )

    # file_list
    file_list = input.get_file_list(RESULT_PATH)

    # parallel upload
    pid_list=range(PARALLEL)
    for i in pid_list:
        p=Process(target=upload,args=(file_list,i))
        p.start()


if __name__ == '__main__':
    main()


