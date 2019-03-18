# -*- coding: utf-8 -*-
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
import random
import math
import matplotlib.pyplot as plt
import numpy as np

# MonteCarlo模拟计算 y=x^2,在[0,10]的面积
def MonteCarlo():
    a=0
    b=10
    def f(x):
        return math.pow(x,2)

    #积分
    s = (math.pow(b, 3) - math.pow(a, 3)) / 3  # 积分公式
    print('s=', s)
    # MonteCarlo 模拟
    def mc():
        t=0
        n=10000
        sum_ = 0
        while t<n-1:
            t=t+1
            x = random.uniform(0, 1) * b  # x的区间是[0,b]
            sum_ = sum_+f(x)

        return sum_/n
    # 面积=长*高
    s=mc()*(b-a)
    print('s=',s)  # 可以发现mc近似sum_



# 对指数分布进行逆变换采样。
# 概率密度：a*exp(-ax)
# 指数分布的累计分布：1-exp(-ax)
# 反函数： x=[-log(1-u)]/a
def Inverse_sample():
    a=1
    def density(x):
        return a* math.exp(-a*x)

    # 逆变换抽样
    # x=[-log(1-u)]/a
    def sample_():
        u = random.uniform(0, 1)
        x = -math.log(1-u)/a
        return x

    x = np.linspace(0, 10, 100)
    y = [density(xi) for xi in x]
    plt.plot(x, y, label='density(x)')
    # 采样10000个点，并统计分布是否跟密度函数曲线是否吻合
    samples = []
    for i in range(10000):
        samples.append(sample_())
    plt.hist(samples, bins=50, normed=True, label='sampling')
    plt.legend()
    plt.show()



# 假设样本点的分布是f(x)=x^2, x的区间是：[0,10]，则它的概率密度函数是：p(x)=x^2/sum_
# 而累计分布无法计算出来，无法进行逆变换采样，因此可以考虑接受-拒绝采样
# 接受-拒绝采样
def Reject_sample():
    def f(x):
        return math.pow(x,2)
    # 计算积分,区间[a,b]
    b=10
    a=0
    sum_ = (math.pow(b, 3) - math.pow(a, 3)) / 3 #积分公式
    print('sum=',sum_)
    # 目标分布，概率分布
    def p(x):
        return f(x)/sum_

    # 参考分布,
    def q(x):
        y=1/(b-a) # 假设参考分布在[a,b]之间是均匀分布
        return y

    # 常数值c
    c = 10
    # 接受-拒绝采样
    def sample_():
        while True:
            x = random.uniform(0, 1)*b  #x的区间是[0,b]
            # 接受-拒绝
            u = random.uniform(0, 1)  #辅助的均匀分布u(0,1)
            if c*u<= p(x)/q(x) :
                return x

    #目标概率密度函数曲线
    x = np.linspace(a, b, 100)
    y = [p(xi) for xi in x]
    plt.plot(x, y,label='p(x)')
    #采样10000个点，并统计分布是否跟密度函数曲线是否吻合
    samples = []
    for  i in range(10000):
        samples.append(sample_())
    plt.hist(samples, bins=50, normed=True,label='sampling')
    plt.legend()
    plt.show()




# 目标：目标的概率分布是概率密度函数是：p(x)=x^2/sum_ ，计划采样5000个点
# 假设选择的转移矩阵 Q(i,j) 的条件转移概率是以i为均值,方差1的正态分布在位置j的值。
# 选择的初始状态值x=[0]，--因为markov链跟初始状态概率分布无关，所以，不需要考虑初始状态概率分布，而且初始值随便选。
def mcmc_sample():
    def f(x):
        return math.pow(x,2)
    # 计算积分,区间[a,b]
    b=1
    a=0
    sum_ = (math.pow(b, 3) - math.pow(a, 3)) / 3 #积分公式
    print(sum_)

    # 目标分布的概率密度是：--注意要限定x的范围
    def p(x):
        if x>0 and x<=b:
            return f(x)/sum_
        else:
            return 0

    # 假设转移矩阵 Q(i,j) 的条件转移概率是以i为均值,方差1的正态分布在位置j的值。任意其实都可以。
    def Q(i,j):
        norm.pdf(j, loc=i,scale=1)

    # 根据Q(i,j)进行采样,
    def q(i):
        # 可以直接使用norm.rvs
        j = norm.rvs(loc=i, scale=1, size=1, random_state=None)
        return j[0] #j是个list,取j[0]就好
    #
    n=6000 #设置采样n次
    x=[0.1]  # 初始状态值=0.1，
    t = 0
    while t < n-1:
        # 从条件概率分布 Q(x|xt) 中采样得到样本x_*
        x_star = q(x[t])
        # print('x_star',x_star)
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
    y=[p(xi) for xi in x]
    plt.scatter(x, y, label='p(x)')  # 散点图
    plt.hist(x, bins=50, normed=1, label='sampling',facecolor='red',alpha=0.7 )
    plt.legend()
    plt.show()

def main():
    mcmc_sample()

if __name__ == '__main__':
    main()