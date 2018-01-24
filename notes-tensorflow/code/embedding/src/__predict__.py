# -*- coding:utf-8 -*-
import os,sys


def get_filepath():
    path = sys.path[0]
    if os.path.isdir(path):
        return path
    elif os.path.isfile(path):
        return os.path.dirname(path)
local_path=get_filepath()
print("change dir:%s" %local_path)
os.chdir(local_path)

#-----------------------------------------
from log import logger
import predict
import upload

def main():
    #
    logger.info("start run predict.py")
    predict.main()
    #
    logger.info("start run upload.py")
    upload.main()


if __name__ == '__main__':
    try:
        main()
    except:
        logger.exception('exception')