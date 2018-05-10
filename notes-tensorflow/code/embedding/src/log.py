# -*- coding:utf-8 -*-
import os
import  logging

log_path ="../logs"
if not os.path.exists(log_path):
    os.mkdir(log_path)


def get_date():
    from datetime import datetime
    now=datetime.now()
    return now.strftime("%Y%m%d")

def get_timestamp():
    import time
    return str(int(time.time()*1000000))

def get_logger(logfile ,logger_name=None):
    # 创建一个logger
    logger= logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    # 创建一个handler，用于写入日志文件
    fh = logging.FileHandler(filename=logfile, mode='a')
    fh.setLevel(logging.INFO)
    #
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # 定义输出格式
    format = "%(asctime)s - %(filename)s[line:%(lineno)d]: %(message)s"
    datefmt= "%d %b %Y %H:%M:%S"
    formatter = logging.Formatter(format,datefmt)
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # 将handler添加到logger里
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


def getLogger():
    logfile_name="sku2vec." + get_timestamp() +".log"
    logfile = os.path.join(log_path,logfile_name)
    get_logger(logfile)