# -*- coding: UTF-8 -*-
import tensorflow as tf
import numpy as np
import  six.moves as s


def build_Model(x,y):
    w = tf.Variable(tf.constant(0.0))

    global_steps = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
    learning_rate = tf.train.exponential_decay(0.01, global_steps, 2, 0.9)
    loss = tf.pow(w*x-y, 2)
    opt=tf.train.GradientDescentOptimizer(learning_rate)
    train_step = opt.minimize(loss, global_step=global_steps)
    return global_steps,learning_rate,train_step

def xavier_init(fan_in, fan_out):
    n = (fan_in + fan_out) / 2.0
    factor=1
    limit = np.sqrt(3.0 * factor / n)
    return tf.random_uniform((fan_in, fan_out),
                             -limit,  limit,
                             dtype = tf.float32)

def train():
    #输入神经元个数
    m=100
    x = np.random.randn(m)
    b = 0
    #隐藏层神经元个数
    n = 1000
    #输出值的方差会随着输入样本的数量而增加。
    w=np.random.randn(n,m)
    #xavier分布
    #w=np.random.randn(m,n)/np.sqrt(m)
    z=np.dot(w,x)+b
    print ('z 均值：', np.mean(z))
    print ('z 方差：', np.var(z))
    f=tf.gfile.FastGFile("aaa","r")
    f.read

def test(n):
    import math
    pos = lambda k: k*(k+1)/2
    i= int(math.sqrt(n))
    while pos(i) < n:
        i += 1
    print(i)

def test_set():
    a=[1,2,4,2,4,5,7,10,5,5,7,8,9,0,3]
    b={}
    for i in a:
        b[i]=1
    c=list(b.keys())
    c.sort()
    print(c)


def merge_list(a,b):
    i=0  # 统计扫描次数
    j =0
    for m in b:
        for n in range(j,len(a)):
            i=i+1
            if m <a[n]:
                print(str(a ) +"," + str(m))
                a=a[0:n]+ [m] + a[n:]
                j=n+1
                break
            elif m > a[-1]:
                a= a + [m]

        print(i)
    return a

# c=merge_list([1,3,7],[2,4,5,6,8])
# print(c)

if __name__ == '__main__':
    test_set()
