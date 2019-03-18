# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
import os
import time
from scipy.misc import imsave
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('../data/MNIST_data/', one_hot=True)

try:
    xrange = xrange
except:
    xrange = range


#--------------------------------------------
flags = tf.app.flags
flags.DEFINE_integer("batch_size", 100 , "batch_size" )
flags.DEFINE_integer("if_train", True , "if_Train=True then train else predict" )
FLAGS = flags.FLAGS

#-----------------------------------------
class Params(object):
    def __init__(self):
        # model params
        self.if_train=FLAGS.if_train
        self.batch_size=FLAGS.batch_size
        self.learning_rate=0.0002
        self.beta1 = 0.5  # momentum term for adam
        self.img_dim = [28,28,1]  # the size of image
        self.y_dim = 10
        self.z_dim = 100  # the dimension of noise z
        self.epoch = 50  # the number of max epoch


class GAN(object):
  def __init__(self, sess, params):
    self.model_path = "../models/"
    self.model_name = "gdn_model.ckpt"
    self.img_path ="../data/images/"
    self.if_train = params.if_train
    self.batch_size = params.batch_size  # must be even number
    self.learning_rate = params.learning_rate
    self.beta1 = params.beta1
    self.img_dim = params.img_dim
    self.z_dim = params.z_dim  # the dimension of noise z
    self.y_dim = params.y_dim
    self.epoch = params.epoch  # the number of max epoch

    self.sess = sess
    self.build_model()  # initializer
    #
    if not os.path.exists(self.model_path):
      os.mkdir(self.model_path)
    if not os.path.exists(self.img_path):
      os.mkdir(self.img_path)


  def leaky_relu(self,x, leakiness=0.2,name='leaky_relu'):
    """Relu, with optional leaky support."""
    return tf.maximum(x, x * leakiness, name=name)

  def flatten(self,x):
    x_shape = x.get_shape()
    x_rank = len(x_shape)  # x.get_shape().ndims
    if (x_rank is None) or (x_rank < 2):
      raise ValueError('Inputs must have a least 2 dimensions.')
    shp = x_shape.as_list()
    flattened_shape = np.prod(shp[1:x_rank])
    resh1 = tf.reshape(x, [-1, flattened_shape])
    return resh1

  def dis_dnn(self,x_image, training=False):
    x=tf.reshape(x_image, (-1, 28*28))
    x =tf.layers.dense(x,256,activation=tf.nn.relu,name="fc1")
    # x = tf.layers.dense(x, 128, activation=tf.nn.relu, name="fc2")
    x = tf.layers.dense(x, 1, activation=None, name="fc3")
    return x

  def gen_dnn(self, noise, training=False ):
    # x = tf.layers.dense(noise, 128, activation=tf.nn.relu, name="fc1")
    x = tf.layers.dense(noise, 256, activation=tf.nn.relu, name="fc2")
    x = tf.layers.dense(x, 28*28, activation=tf.nn.relu, name="fc3")
    x = tf.reshape(x,[-1,28,28,1])
    return x


  def discriminator(self, inputs, reuse,if_train=False):
    """
    :param inputs: shape=[None,h,w,c]
    """
    with tf.variable_scope("dis", reuse=reuse):
      outputs=self.dis_dnn(inputs,if_train)
      logits=tf.nn.sigmoid(outputs )
    return logits

  def generator(self, noise, reuse=False, if_train=False):
    """
    :return: tensor = [None,h,w,c]
    """
    with tf.variable_scope("gen", reuse=reuse):
      x=self.gen_dnn(noise,training=if_train)
      return x


  def build_model(self):
    # y 表示label
    # self.label = tf.placeholder(tf.float32, shape=[None, self.y_dim], name='label')
    # x_img表示真实图片
    self.x = tf.placeholder(tf.float32, shape=[None, self.img_dim[0],self.img_dim[1], self.img_dim[2]],
                            name='real_img')
    # z 表示随机噪声
    self.z = tf.placeholder(tf.float32, shape=[None, self.z_dim], name='noise')

    # 由生成器生成图像 G
    self.G_img = self.generator(self.z,reuse=False,if_train=self.if_train ) #shape=[batch_size,28,28,1]
    # 真实图像送入判别器
    d_logits_r = self.discriminator(self.x, reuse=False,if_train=self.if_train)
    # 生成图像送入辨别器
    d_logits_f = self.discriminator(self.G_img, reuse=True,if_train=self.if_train)

    # loss
    self.d_loss, self.g_loss = self.get_loss(d_logits_r, d_logits_f)
    #
    self.train_op_d, self.train_op_g = self.get_train_op(self.d_loss, self.g_loss)

    self.saver = tf.train.Saver()
    #
    if self.if_train:
      init = tf.global_variables_initializer()
      self.sess.run(init)
    else:
      ckpt = tf.train.get_checkpoint_state(self.model_path)
      self.saver.restore(self.sess, ckpt.model_checkpoint_path)
      print('model load done')

  def get_loss(self, d_logits_r, d_logits_f):
    d_loss_r = tf.reduce_mean(tf.log(tf.clip_by_value(d_logits_r, 1e-10, 1.0)))
    d_loss_f = tf.reduce_mean(tf.log(tf.clip_by_value(1.0 - d_logits_f, 1e-10, 1.0)))
    d_loss = -(d_loss_r + d_loss_f)
    g_loss = -tf.reduce_mean(tf.log(tf.clip_by_value(d_logits_f, 1e-10, 1.0)))
    return d_loss, g_loss

  def get_train_op(self,d_loss,g_loss):
    all_vars = tf.trainable_variables()
    g_vars = [v for v in all_vars if 'gen' in v.name]
    d_vars = [v for v in all_vars if 'dis' in v.name]
    for v in all_vars:
      print(v)

    opt_d = tf.train.AdamOptimizer(self.learning_rate ,beta1=0.5)
    opt_g = tf.train.AdamOptimizer(self.learning_rate ,beta1=0.5)
    train_op_d = opt_d.minimize(d_loss, var_list=d_vars)
    train_op_g = opt_g.minimize(g_loss, var_list=g_vars)
    return train_op_d,train_op_g

  def train(self):
    for epoch_i in range(self.epoch):
      iters = int(50000 / self.batch_size)
      for step in range(iters):
        t1 = time.time()
        # input
        with tf.device('/cpu:1'):
          batch_x, batch_y = mnist.train.next_batch(self.batch_size) #x.shape=[100,784]
          batch_x = batch_x.reshape(self.batch_size,self.img_dim[0],self.img_dim[1],self.img_dim[2])
          batch_z = np.random.uniform(-1.0, 1.0, [self.batch_size, self.z_dim])

        # update the Discrimater k times
        _, loss_d, = self.sess.run([self.train_op_d,self.d_loss],
                                   feed_dict={self.x:batch_x, self.z: batch_z})
        # update the Generator one time
        _, loss_g, = self.sess.run([self.train_op_g, self.g_loss],
                                   feed_dict={self.x: batch_x, self.z: batch_z})

        if (step + 1) % 100 == 0 :
          t2 = time.time()
          print("[cost %3f][epoch:%2d/%2d][iter:%4d/%4d],loss_d:%5f,loss_g:%4f "
                % (t2 - t1, epoch_i, self.epoch, step+1, iters, loss_d, loss_g))

        dispaly_step = 500
        if (step + 1) % dispaly_step == 0:
          img_name = "sample{}_{}.jpg".format(epoch_i, (step+1)// dispaly_step)
          print("saving image: %s"%(img_name))
          self.save_images(os.path.join(self.img_path ,img_name ), if_transform=False)

      # 每隔n=1 epoch,保存模型
      # if (epoch_i + 1) % 1 == 0:
      #   checkpoint_path = os.path.join(self.model_path, self.model_name )
      #   self.saver.save(self.sess, checkpoint_path, global_step=epoch_i)



  def save_images(self, save_path, if_transform = False):
    k=10
    test_z = np.random.uniform(-1.0, 1.0, size=[k ** 2, self.z_dim])
    x = self.sess.run(self.G_img, feed_dict={self.z: test_z})
    x=np.squeeze(x)
    # x = np.random.uniform(-1, 1, [100, 28, 28, 1])

    # 数值从[-1, 1] -> [0,255]
    if if_transform:
      x = (255.0*(x +1)/2)#.astype('uint8')
    h,w,c=self.img_dim[0],self.img_dim[1],self.img_dim[2]
    if c==1:
      img = np.zeros((h * k, w * k))
    else:
      img = np.zeros((h * k, w * k,3))
    for i in range(k):
      for j in range(k):
        n=i * k +j
        img[h * i:h * (i + 1), w * j:w * (j + 1)] = x[n]
    imsave(save_path, img)


# ------------------------------------------------------------------------
CONFIG = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
CONFIG.gpu_options.allow_growth = True

def main():

  sess = tf.InteractiveSession(config=CONFIG)
  params =Params()
  model = GAN(sess, params)
  model.train()
  # model.save_images("../data/images/test.jpg",True)

if __name__ == '__main__':
  main()