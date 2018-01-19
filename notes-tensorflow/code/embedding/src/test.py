# -*- coding: utf-8 -*-

import heapq
import numpy as np
import time
from multiprocessing.pool import ThreadPool




def main():
    pool = ThreadPool(4)
    x =np.random.uniform(0,10,5000000)
    t1=time.time()

    def find_top(k):
        return np.argsort(x)[0:5]

    # for i in range(100):
        # heapq.nsmallest(10, x, key=None)
        # x.argsort()[0:5] # 等同np.argsort(x)
        # np.argsort(x)[0:5]
    pool.map(find_top, range(10) )
    t2=time.time()
    print("cost %s sec" %(t2-t1))



if __name__ == '__main__':
    main()
    # a = [1,2,35,32,6534,6,314,325,246,]
    # for i in map(printt, a, range(9)):
    #     print(i)