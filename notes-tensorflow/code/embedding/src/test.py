# -*- coding:utf-8 -*-
from threading import Thread
from multiprocessing import Process
import time
import os,sys

def my_counter():
    i = 0
    for _ in range(100000000):
        i = i + 1
    return True

def single_thread():
    start_time = time.time()
    for tid in range(2):
        t = Thread(target=my_counter)
        t.start()
        t.join()
    end_time = time.time()
    print("Total time: {}".format(end_time - start_time))

def multi_thread():
    thread_array = {}
    start_time = time.time()
    for tid in range(2):
        t = Thread(target=my_counter)
        t.start()
        thread_array[tid] = t
    for i in range(2):
        thread_array[i].join()
    end_time = time.time()
    print("Total time: {}".format(end_time - start_time))

def single_process():
    start_time = time.time()
    for tid in range(2):
        t = Process(target=my_counter)
        t.start()
        t.join()
    end_time = time.time()
    print("Total time: {}".format(end_time - start_time))


def multi_process():
    process_list = []
    start_time = time.time()
    for tid in range(2):
        p = Process(target=my_counter)
        p.start()
        process_list.append(p)
    for p in process_list:
        p.join()
    end_time = time.time()
    print("Total time: {}".format(end_time - start_time))


# 返回到顶层目录
def get_filepath():
    # path =__file__
    path = sys.path[0]
    if os.path.isdir(path):
        return os.path.dirname(path)
    elif os.path.isfile(path):
        return os.path.dirname(os.path.dirname(path))


def chdir_cur():
    abspath = os.path.dirname(__file__)
    print(abspath)
    # sys.path.append(abspath)
    if abspath=='':
        print('1' +sys.path[0])
    else:
        print('2' +abspath)


def main():
    import logging
    logging.info("test logging ")
    print("test print")
    # multi_process()


if __name__ == '__main__':
    main()
