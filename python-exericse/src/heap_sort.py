# -*- coding: utf-8 -*-
import time
import numpy as np


class Heap():
    """
    大顶堆
    """
    def __init__(self,L):
        """初始化，需要传入一个List作为Heap的初始值"""
        self.heap=L
        self.length=len(L)
        self.sort()

    def add(self,elem):
        """
        1. 如果取 max top n ,则elem > min(L)时进行add
        2. 如果取 min top n ,则elem < max(L)时进行add
        """
        if elem<self.heap[-1]:
            self.heap[-1]=elem
            self.sort()
            # print(self.heap)

    def sort(self):
        # self.heap=np.sort(self.heap,kind="heapsort")
        self.heapsort()

    def adjust(self,start_idx, end_idx):
        """
        从start_id开始，保证一条线的子堆是符合要求的，父节点大于他们的子节点
        """
        # 第一个节点的index开始
        parent = start_idx
        child = parent *2 +1  # 左孩子

        while child < end_idx:
            # 如果有右孩子结点，并且右孩子结点的值大于左孩子结点，则选取右孩子结点
            if (child+1 < end_idx) and (self.heap[child] < self.heap[child+1]):
                child += 1
            # 如果父结点的值<孩子结点的值，则交换
            if self.heap[parent] < self.heap[child]:
                temp = self.heap[parent]
                self.heap[parent] = self.heap[child]
                self.heap[child] =temp
                # 选取孩子结点的左孩子结点,继续向下筛选
                parent = child
                child = 2 * parent +1
            else: # 如果父结点的值已经大于孩子结点的值，则直接结束
                break

    def firstHeap(self):
        """
        bulid the first heap
        """
        n = self.length - 1
        start = n // 2  # 从最下面最左边第一个枝节点开始
        while start >= 0:
            self.adjust(start, n)
            start -= 1


    def heapsort(self):
        self.firstHeap()

        # 对剩余的节点进行排序
        j= self.length-1
        start =0
        while j>0 :
            #将最大的元素放在最后，进行排除。
            temp = self.heap[0]
            self.heap[0] = self.heap[j]
            self.heap[j] = temp
            # 筛选 R[0] 结点，得到i-1个结点的堆
            self.adjust( start, j)
            #
            j -=1



def main():
    # x= np.array([1, 3, 4, 5, 2, 6, 9, 7, 8, 0])
    x = np.random.uniform(0, 10, 5000000)
    n=10
    a =x[0:n]  #建立一个只有10长度的数组
    heap = Heap(a)

    t1 = time.time()
    # 添加剩余的元素，并进行比较
    for i in x[n:]:
        heap.add(i)
    print(heap.heap)
    t2 = time.time()
    print("cost %s sec" % (t2 - t1))
    #
    # np.sort(x,kind="heapsort")[0:5]
    # t3 = time.time()
    # print("cost %s sec" % (t3 - t2))



if __name__ == '__main__':
    main()