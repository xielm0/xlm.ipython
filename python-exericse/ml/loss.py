# -*- coding:utf-8 -*-
import numpy as np
from matplotlib import pyplot as plt


def Loss01(yf):
    if yf>=0 :
        return 0
    else:
        return 1

def HingeLoss(yf):
    return np.maximum(0,1-yf)

def LogLoss(yf):
    return np.log2(1+np.exp(-1.0 * yf))

def EntropyLoss(yf):
    return -1.0 * np.log2( (1+yf)/2)

#
x=list(np.arange(-2,2,0.01))
# # 定义坐标轴
# plt.xlabel('yf')     # x轴标签
# plt.ylabel('loss')
# plt.xlim(-2.0, 2.0)  # x轴的范围
# plt.ylim(0.0, 4.)
# #plot
# plt.plot( x,list(map(Loss01,x)),'k')
# plt.plot( x,list(map(HingeLoss,x)),'r')
# plt.plot( x,list(map(LogLoss,x)),'g')
# plt.plot( x,list(map(EntropyLoss,x)),'b')
#
plt.show()


def L2Loss(a):
    return np.square(a)

def L1Loss(a):
    return np.abs(a)

def huber(a):
    delta=0.8
    if np.abs(a)<=delta:
        return L2Loss(a)
    else:
        return (2* L1Loss(a)-delta) * delta

# a=list(np.arange(-2,2,0.01))
# # 定义坐标轴
# plt.xlabel('a')     # x轴标签
# plt.ylabel('loss')
# plt.xlim(-2.0, 2.0)  # x轴的范围
# plt.ylim(0.0, 4.)
# #plot
# plt.plot( a,L2Loss(a),'g')
# plt.plot( a,L1Loss(a),'b')
# plt.plot( a,list(map(huber,a)),'r')
#
# plt.show()