# -*- coding:utf-8 -*-
import os,sys

def cd_cur_dir():
    abspath = os.path.dirname(__file__)
    if abspath=='':
        path =sys.path[0]
    else:
        path = abspath
    # change dir
    print("change dir:%s" %path)
    os.chdir(path)

cd_cur_dir()
#-----------------------------------------
import download
import predict
import upload
import log


def main():
    #
    logger=log.getLogger()
    #
    download.download_apply()
    #
    predict.main()
    #
    upload.main()


if __name__ == '__main__':
    main()