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
import train
import download
import input


def main():
    #
    logger.info("start run download.py")
    download.main()
    #
    logger.info("start run download.py")
    input.gen_train_tf()
    #
    logger.info("start run train.py")
    train.main()


if __name__ == '__main__':
    try:
        main()
    except:
        logger.exception('exception')
