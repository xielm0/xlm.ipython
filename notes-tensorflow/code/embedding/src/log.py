# -*- coding:utf-8 -*-
import os
import  logging

log_path ="../logs"
def get_logger(logfile_name="sku2vec.log"):
    # 创建一个logger
    logger= logging.getLogger()
    logger.setLevel(logging.INFO)
    # 创建一个handler，用于写入日志文件
    logfile = os.path.join(log_path,logfile_name)
    # print(filename)
    fh = logging.FileHandler(filename=logfile, mode='a')
    fh.setFormatter(logging.INFO)
    #
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # 定义输出格式
    format = "%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s"
    datefmt= "%a, %d %b %Y %H:%M:%S"
    formatter = logging.Formatter(format,datefmt)
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # 将handler添加到logger里
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger

logger=get_logger()