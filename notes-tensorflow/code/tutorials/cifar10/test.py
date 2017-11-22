# -*- coding: UTF-8 -*-
import tensorflow as tf
import numpy as np


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
    m=1000
    x = np.random.randn(m).T
    b = 0
    #隐藏层神经元个数
    n = 10000
    #输出值的方差会随着输入样本的数量而增加。
    w=np.random.randn(m,n)
    #xavier分布
    #w=np.random.randn(m,n)/np.sqrt(m)
    z=np.dot(w,x)+b
    print ('z 均值：', np.mean(z))
    print ('z 方差：', np.var(z))


if __name__ == '__main__':
    train()

