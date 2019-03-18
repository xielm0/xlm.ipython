# -*- coding: utf-8 -*-
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt

def markov_test():
    p = np.array([[0.6,0.2,0.2],[0.3,0.4,0.3],[0,0.3,0.7]],dtype='float32')
    x = np.array([[0.5,0.3,0.2]],dtype='float32') # 一个初始分布。sum=1
    # x = np.array([[0.2,0.3,0.5]],dtype='float32') #换一个初始分布

    value1 = []
    value2 = []
    value3 = []
    for i in range(30):
        x = np.dot(x,p)
        value1.append(x[0][0])
        value2.append(x[0][1])
        value3.append(x[0][2])
    print(x)  # 最终的分布值=[[ 0.23076934  0.30769244  0.46153861]]
    t = np.arange(30)
    plt.plot(t,value1,label='x(1)')
    plt.plot(t,value2,label='x(2)')
    plt.plot(t,value3,label='x(3)')
    plt.legend()
    plt.show()

# 目标：目标的概率分布是一个均值3，标准差2的正态分布，计划抽样5000个样本,
# 假设选择的转移矩阵 Q(i,j) 的条件转移概率是以i为均值,方差1的正态分布在位置j的值。
# 选择的初始状态值x=[0]，--因为markov链跟初始状态概率分布无关，所以，不需要考虑初始状态概率分布，而且初始值随便选。
def mcmc_sample():
    # 目标分布的概率密度是：
    def p(x):
        #pdf: Probability density function
        y = norm.pdf(x, loc=3, scale=2)
        return y

    # 假设转移矩阵 Q(i,j) 的条件转移概率是以i为均值,方差1的正态分布在位置j的值。其他的也可以。
    def Q(i,j):
        norm.pdf(j, loc=i,scale=1)

    # 根据Q(i,j)进行采样,
    def q(i): 
        # 可以直接使用norm.rvs
        j = norm.rvs(loc=i, scale=1, size=1, random_state=None)
        return j[0] #j是个list,取j[0]就好
    #
    n=6000 #设置采样n次
    x=[0]  # 初始状态值=0
    t = 0
    while t < n-1:
        # 从条件概率分布 Q(x|xt) 中采样得到样本x_*
        x_star = q(x[t])
        # 计算alpha值
        alpha = min(1, (p(x_star) / p(x[t])))
        # 从均匀分布采样u∼uniform[0,1]
        u = np.random.uniform(0, 1)
        if u < alpha:
            x.append(x_star)  #接受采样
        else:
            x.append(x[t])

        t=t+1

    #假设1000次采样后进入平稳概率分布，即平稳概率分布等于目标的概率分布。
    # 也就是1000次后，是按照目标概率分布进行接受-拒绝采样。
    x=x[1000:]
    plt.scatter(x, p(x),label='p(x)')  #散点图
    plt.hist(x, bins=50, normed=1, label='sampling',facecolor='red',alpha=0.7 )
    plt.legend()
    plt.show()

def main():
    mcmc_sample()

if __name__ == '__main__':
    main()
