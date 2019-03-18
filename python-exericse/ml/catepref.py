# -*- coding: UTF-8 -*-

import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.model_selection  import train_test_split
import logging
import os
from sklearn import metrics

flags = tf.app.flags
flags.DEFINE_integer("batch_size", 10 , "batch_size" )

FLAGS = flags.FLAGS


# model
MODEL_SAVE_PATH = "../models/"
MODEL_NAME = "model.ckpt"
learning_rate=0.0001
batch_size=100
cate_num=1000

def get_loss(x, y_,  scope):
    y = inference(x, train_flag=True)
    #cross_entropy =tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y, labels=y_))
    cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)) + (1- y_) * tf.log(tf.clip_by_value(1-y, 1e-10, 1.0)))
    # scope="GPU_i" ,so计算当前GPU上的loss
    regularization_loss = tf.add_n(tf.get_collection('losses', scope))
    loss = cross_entropy + regularization_loss
    #
    with tf.name_scope("loss"):
        tf.summary.scalar("cross_entropy", cross_entropy)
        tf.summary.scalar("regularization_loss", regularization_loss)
        tf.summary.scalar("loss", loss)

    return loss


F=tf.placeholder(tf.float32,[batch_size,cate_num],"feature")
Y=tf.placeholder(tf,[batch_size,1],"feature")

W=tf.get_variable("w",[cate_num,cate_num]
                  ,tf.random_uniform_initializer(-0.001,0.001)
                  ,dtype=tf.float32)



S=tf.multiply(F,W)
loss=get_loss(F,S)
opt=tf.train.GradientDescentOptimizer(learning_rate)
train_op=opt.minimize(loss)

with tf.Session() as sess:
    sess.run(train_op)








if __name__ == '__main__':
    train()
    eval()