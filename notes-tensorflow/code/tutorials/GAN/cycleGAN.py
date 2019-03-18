# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
import os
import time
from PIL import Image
from scipy.misc import imsave

try:
    xrange = xrange
except:
    xrange = range


CONFIG = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
CONFIG.gpu_options.allow_growth = True
#--------------------------------------------
flags = tf.app.flags
flags.DEFINE_integer("batch_size", 1 , "batch_size" )
flags.DEFINE_bool("if_train", True , "if_Train=True then train else predict" )
flags.DEFINE_bool("gen_TFrecords", False , "gen_TFrecords" )
FLAGS = flags.FLAGS
gen_TFrecords=FLAGS.gen_TFrecords
#-----------------------------------------
def imread(img_path):
  img = Image.open(img_path)
  img = np.array(img)
  return img

class DataSet(object):

  def __init__(self, x, y) :
    self._x = x
    self._y = y
    self._epochs_completed = 0
    self._index_in_epoch = 0
    self._num_examples = x.shape[0]

  def next_batch(self, batch_size,shuffle=True):
    start = self._index_in_epoch
    # Shuffle for the first epoch
    if self._epochs_completed == 0 and start == 0 and shuffle:
      perm0 = np.arange(self._num_examples)
      np.random.shuffle(perm0)
      self._x = self._x[perm0]
      self._y = self._y[perm0]
    # Go to the next epoch
    if start + batch_size > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Get the rest examples in this epoch
      rest_num_examples = self._num_examples - start
      x_rest_part = self._x[start:self._num_examples]
      y_rest_part = self._y[start:self._num_examples]
      # Shuffle the data
      if shuffle:
        perm = np.arange(self._num_examples)
        np.random.shuffle(perm)
        self._x = self._x[perm]
        self._y = self._y[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size - rest_num_examples
      end = self._index_in_epoch
      x_new_part = self._x[start:end]
      y_new_part = self._y[start:end]
      return np.concatenate((x_rest_part, x_new_part), axis=0), np.concatenate((y_rest_part, y_new_part), axis=0)
    else:
      self._index_in_epoch += batch_size
      end = self._index_in_epoch
      return self._x[start:end], self._y[start:end]

class Input_data():
  """
  将图片转成tfrecords格式，再训练
  """
  def __init__(self):
      self.data_dir ="../data/apple2orange"

  def read_images(self,dir_path):
    images = os.listdir(dir_path)
    images = filter(lambda x: x.endswith(".jpg"),images)
    res =[]
    for img_file in images:
      tmp =imread(os.path.join(dir_path, img_file))
      res.append(tmp)
    return res

  def read_data_sets(self,dir_path):
    #download
    # source_path = "https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/apple2orange.zip"
    trainA = self.read_images(os.path.join(dir_path, 'trainA'))
    trainB = self.read_images(os.path.join(dir_path, 'trainB'))
    # testA = read_images(os.path.join(dir_path, 'testA'))
    # testB = read_images(os.path.join(dir_path, 'testB'))
    train=DataSet(trainA, trainB )
    # test =DataSet(testA, testB )
    return train

  def get_train_data(self):
    train = self.read_data_sets("../data/apple2orange" )
    return train

  def _int64_feature(self,value):
      return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
  def _bytes_feature(self,value):
      return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

  def images2TFrecord(self,dir_path,TFrecord_name):
    images = os.listdir(dir_path)
    images = filter(lambda x: x.endswith(".jpg"),images)
    writer = tf.python_io.TFRecordWriter(TFrecord_name)
    for img_file in images:
      image = imread(os.path.join(dir_path, img_file))
      image_raw = image.tobytes()
      example = tf.train.Example(features=tf.train.Features(feature={
        'image_raw': self._bytes_feature(image_raw)
        }))
      writer.write(example.SerializeToString())
    writer.close()


  def read_TFrecord(self,TFrecord_name,batch_size,epochs=1):
      filename_queue = tf.train.string_input_producer([TFrecord_name], num_epochs=epochs)
      reader = tf.TFRecordReader()
      _, serialized_example = reader.read_up_to(filename_queue,100)

      # 使用 read_up_to ，则使用 parse_example ，否则使用 parse_single_example
      features = tf.parse_example(
          serialized_example,
          features={
              'image_raw': tf.FixedLenFeature([], tf.string),
          })
      # 将string转换为数字
      decoded_image = tf.cast(tf.decode_raw(features['image_raw'], tf.uint8),tf.float32)
      image = tf.reshape(decoded_image, [-1,256,256,3])
      # 并且转换到[-1,1]
      image = (image/127.5) -1.0

      min_after_dequeue = 200
      capacity = min_after_dequeue + 3 * batch_size
      batch_data =tf.train.shuffle_batch(
          [image ],
          batch_size=batch_size,
          capacity=capacity,
          min_after_dequeue=min_after_dequeue,
          num_threads=10, enqueue_many=True
          )
      return batch_data


input_data =Input_data()
data_path=input_data.data_dir
# train = input_data.get_train_data()

if gen_TFrecords:
  input_data.images2TFrecord("../data/apple2orange/trainA",os.path.join(data_path,"trainA.tfrecords"))
  input_data.images2TFrecord("../data/apple2orange/trainB",os.path.join(data_path,"trainB.tfrecords"))


class Params(object):
    def __init__(self):
        # model params
        self.if_train=FLAGS.if_train
        self.batch_size=FLAGS.batch_size
        self.learning_rate=0.0002
        self.beta1 = 0.5  # momentum term for adam
        self.img_dim = [256,256,3]  # the size of image
        self.epoch = 100  # the number of max epoch


class GAN(object):
  def __init__(self, params):
    self.model_path = "../models/"
    self.model_name = "gdn_model.ckpt"
    self.img_path ="../data/images/"
    self.tensorboard_path = "../logs/tensorboard/"
    self.if_train = params.if_train
    self.batch_size = params.batch_size  # must be even number
    self.learning_rate = params.learning_rate
    self.lambda1 = 10
    self.beta1 = params.beta1
    self.img_dim = params.img_dim
    self.epoch = params.epoch  # the number of max epoch
    self.use_FCN = False
    self.use_sigmoid =False

    #
    if not os.path.exists(self.model_path):
      os.mkdir(self.model_path)
    if not os.path.exists(self.img_path):
      os.mkdir(self.img_path)
    if not os.path.exists(self.tensorboard_path):
      os.mkdir(self.tensorboard_path)
    elif self.if_train:
      os.system("/bin/rm  " + os.path.join(self.tensorboard_path,"events.out.tfevents.*"))


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

  def instance_norm(self,x,name="in",training=False):
    with tf.variable_scope(name):
      x_shape = x.shape.as_list()
      params_shape = x_shape[-1]
      beta = tf.get_variable('beta', params_shape, initializer=tf.zeros_initializer())
      gamma = tf.get_variable('gamma', params_shape, initializer=tf.ones_initializer())
      epsilon = 1e-9
      # batch_norm是axes=[0,1,2], instacne_norm是axes=[1,2]
      mean, var = tf.nn.moments(x, [1, 2], keep_dims=True)
    return gamma*(x-mean)/tf.sqrt(var+epsilon)+beta

  def conv_bn(self,x,name, depth, ksize, stride, training=False,activation_fn=tf.nn.relu):
    with tf.variable_scope(name):
      x = tf.layers.conv2d(x, depth, [ksize, ksize], strides=(stride, stride), padding='SAME',name="conv1")
      x = self.instance_norm(x, training=training)
      # x = tf.layers.batch_normalization(x, training=training)
      x = activation_fn(x)
    return x

  def deconv_bn(self,x,name, depth, ksize, stride, training=False,activation_fn=tf.nn.relu):
    with tf.variable_scope(name):
      x = tf.layers.conv2d_transpose(x, depth, [ksize, ksize], strides=(stride, stride), padding='SAME')
      x = self.instance_norm(x , training=training)
      x = activation_fn(x)
    return x

  def res(self, org_x, name,out_depth,stride=2):
    """
    conv+bn+relu
    """
    activation_fn=tf.nn.relu
    with tf.variable_scope(name):
      mid_depth = out_depth / 4
      with tf.variable_scope('sub1'):
        x = tf.layers.conv2d(org_x, mid_depth, [1, 1], strides=(1, 1), padding='SAME',name="cov1")
        x = self.instance_norm(x)
        x = activation_fn(x)

      with tf.variable_scope('sub2'):
        x = tf.layers.conv2d(x, mid_depth, [3, 3], strides=(stride, stride), padding='SAME', name="cov2")
        x = self.instance_norm(x)
        x = activation_fn(x)

      with tf.variable_scope('sub3'):
        x = tf.layers.conv2d(x, out_depth, [1, 1], strides=(1, 1), padding='SAME', name="cov3")
        x = self.instance_norm(x)
        x = activation_fn(x)

      with tf.variable_scope('sub_add'):
        shortcut = tf.layers.conv2d(org_x, out_depth, [1, 1], strides=(stride, stride), padding='SAME', name="shortcut")
        x += shortcut

      tf.logging.info('image after unit %s', x.get_shape())
      return x


  def discriminator(self,name, x_image, reuse, training=False):
    # x_image.shape=[256,256,3]
    activation_fn = self.leaky_relu
    with tf.variable_scope(name, reuse=reuse):
      outputs = self.conv_bn(x_image, "conv1", 64 , 3, 2, training, activation_fn)  # 128x128x64
      outputs = self.conv_bn(outputs, "conv2", 128, 3, 2, training, activation_fn) # 64x64x128
      outputs = self.conv_bn(outputs, "conv3", 256, 3, 2, training, activation_fn) # 32x32x256
      outputs = self.conv_bn(outputs, "conv4", 512, 3, 2, training, activation_fn) # 16x16x512
      print("last conv.shape=", outputs.shape.as_list())
      # 最后一层，可以采用FC，也可以采用FCN
      with tf.variable_scope('fc'):
        if self.use_FCN:
          outputs = tf.layers.conv2d(outputs, 1, [3, 3], strides=(1, 1), padding='VALID')
        else:
          outputs = self.flatten(outputs)
          outputs = tf.layers.dense(outputs, 1, name='outputs')
        if self.use_sigmoid:
          outputs = tf.sigmoid(outputs)
      print("d.output.shape=",outputs.shape.as_list())
    return outputs


  def generator(self,name, x_image ,reuse,training=False):
    with tf.variable_scope(name, reuse=reuse):
      # 3层卷积(ksize=7,3,3) + 5层 res + 3层反卷积
      # x.shape=[256,256,3]
      x = self.conv_bn(x_image, "conv1", 64, 7, 2, training)  #128
      x = self.conv_bn(x, "conv2", 128, 3, 2, training) #64
      x = self.conv_bn(x, "conv3", 256, 3, 2, training) #32

      #
      x = self.res(x, "res1", 256, 1)
      x = self.res(x, "res2", 256, 1)
      x = self.res(x, "res3", 256, 1)
      x = self.res(x, "res4", 256, 1)
      x = self.res(x, "res5", 256, 1)
      x = self.res(x, "res6", 256, 1)
      x = self.res(x, "res7", 256, 1)
      x = self.res(x, "res8", 256, 1)
      x = self.res(x, "res9", 256, 1)
      #
      x = self.deconv_bn(x, "deconv1", 128, 3, 2, training) #64
      x = self.deconv_bn(x, "deconv2", 64, 3, 2, training)  #128
      x = self.deconv_bn(x, "deconv3", 3, 7, 2, training,activation_fn=tf.nn.tanh)   #256

    return x


  def build_model(self,x,y):
    # x->y
    self.fake_y = self.generator("G",x,reuse=False,training=self.if_train)
    # 真实图像送入判别器
    d_y_r = self.discriminator("D_Y",y, reuse=False,training=self.if_train)
    # 生成图像送入辨别器
    d_y_f = self.discriminator("D_Y",self.fake_y, reuse=True,training=self.if_train)
    # 生成x的还原图像
    x_sim = self.generator("F",self.fake_y, reuse=False, training=self.if_train)

    # y->x
    self.fake_x = self.generator("F",y, reuse=True, training=self.if_train)
    # 真实图像送入判别器
    d_x_r = self.discriminator("D_X",x, reuse=False, training=self.if_train)
    # 生成图像送入辨别器
    d_x_f = self.discriminator("D_X",self.fake_x, reuse=True, training=self.if_train)
    # 生成x的还原图像
    y_sim = self.generator("G",self.fake_x, reuse=True, training=self.if_train)

    # loss
    self.d_y_loss, self.g_gan_loss = self.get_gan_loss(d_y_r,d_y_f)
    self.d_x_loss, self.f_gan_loss = self.get_gan_loss(d_x_r, d_x_f)
    self.cycle_loss1 = self.get_cycle_loss(x, x_sim)
    self.cycle_loss2 = self.get_cycle_loss(y, y_sim)
    self.cycle_loss = self.cycle_loss1+ self.cycle_loss2
    # g_loss += cycle_loss
    self.g_loss = self.g_gan_loss + self.cycle_loss * self.lambda1
    self.f_loss = self.f_gan_loss + self.cycle_loss * self.lambda1

    #
    self.train_op= self.get_train_op(self.d_y_loss,self.g_loss,self.d_x_loss, self.f_loss)
    #
    # return self.train_op

  def get_gan_loss(self, d_logits_r, d_logits_f):
    # d_loss = -(tf.reduce_mean(d_logits_r) - tf.reduce_mean(d_logits_f))
    # g_loss = -tf.reduce_mean(d_logits_f)
    d_loss = tf.reduce_mean(tf.square(d_logits_r -1)) + tf.reduce_mean(tf.square(d_logits_f))
    g_loss = tf.reduce_mean(tf.square(d_logits_f -1))
    return d_loss, g_loss

  def get_cycle_loss(self, x , x_sim):
    loss1 = tf.reduce_mean(tf.abs(x-x_sim))
    return loss1

  def get_train_op(self,d_y_loss,g_loss,d_x_loss,f_loss):
    all_vars = tf.trainable_variables()
    D_Y_vars = [v for v in all_vars if 'D_Y' in v.name]
    G_vars = [v for v in all_vars if 'G' in v.name]
    D_X_vars = [v for v in all_vars if 'D_X' in v.name]
    F_vars = [v for v in all_vars if 'F' in v.name]
    # for v in all_vars:
    #   print(v)

    opt_d_y = tf.train.AdamOptimizer(self.learning_rate, beta1=0.5, beta2=0.9)
    opt_g = tf.train.AdamOptimizer(self.learning_rate, beta1=0.5, beta2=0.9)
    opt_d_x = tf.train.AdamOptimizer(self.learning_rate, beta1=0.5, beta2=0.9)
    opt_f = tf.train.AdamOptimizer(self.learning_rate, beta1=0.5, beta2=0.9)
    #
    train_op_d_y = opt_d_y.minimize(d_y_loss, var_list=D_Y_vars)
    train_op_g = opt_g.minimize(g_loss, var_list=G_vars)
    train_op_d_x = opt_d_x.minimize(d_x_loss, var_list=D_X_vars)
    train_op_f = opt_f.minimize(f_loss, var_list=F_vars)
    train_op = tf.group(train_op_d_y,train_op_g,train_op_d_x,train_op_f)

    return train_op


  def train(self):
    # y = tf.placeholder(tf.float32, shape=[None, self.img_dim[0], self.img_dim[1], self.img_dim[2]], name='y')
    # x = tf.placeholder(tf.float32, shape=[None, self.img_dim[0], self.img_dim[1], self.img_dim[2]], name='x')
    # 从tf-records读取数据
    y = input_data.read_TFrecord(os.path.join(data_path, "trainB.tfrecords"), self.batch_size,self.epoch)
    x = input_data.read_TFrecord(os.path.join(data_path, "trainA.tfrecords"), self.batch_size,self.epoch)
    print("x.shape=",x.shape.as_list())

    #
    self.build_model(x,y)
    #
    merged_summary_op = tf.summary.merge_all()
    saver = tf.train.Saver()
    with tf.Session(config=CONFIG) as sess:
      init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
      sess.run(init)
      #
      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(coord=coord, sess=sess)
      #
      summary_writer = tf.summary.FileWriter(self.tensorboard_path, sess.graph)
      # 测试输入与输出的值的范围
      # print("x_real=", sess.run(x))
      # print("fake_y=", sess.run(self.fake_y))
      # print("fake_x=", sess.run(self.fake_x))
      #
      g_step = 0
      for epoch_i in range(self.epoch):
        # num_examples = train._num_examples
        num_examples = 900
        iters = int(num_examples / self.batch_size)
        for step in range(iters):
          g_step += 1
          t1 = time.time()
          # input
          # batch_x, batch_y = train.next_batch(self.batch_size)  # x.shape=[batch,256,256,3 ]

          _, g_loss,f_loss,d_y_loss,d_x_loss,cycle_loss= sess.run(
            [self.train_op,self.g_loss,self.f_loss,self.d_y_loss,self.d_x_loss,self.cycle_loss]
                                                        # ,feed_dict = {x: batch_x,y:batch_y}
                                                        )

          if (step + 1) % 10 == 0 :
            t2 = time.time()
            print("[cost %3f][epoch:%2d/%2d][iter:%4d/%4d],g_loss:%4f,f_loss:%4f,d_y_loss:%4f,d_x_loss:%4f,cycle_loss:%4f "
                  % (t2 - t1, epoch_i, self.epoch, step+1, iters, g_loss, f_loss, d_y_loss, d_x_loss,cycle_loss))
            summary = sess.run(merged_summary_op )
            summary_writer.add_summary(summary, g_step)


        # 每隔n=1 epoch,保存模型
        if (epoch_i + 1) % 1 == 0:
          checkpoint_path = os.path.join(self.model_path, self.model_name )
          saver.save(sess, checkpoint_path, global_step=epoch_i)

      coord.request_stop()
      coord.join(threads)


  def inference(self,x):
    # x->y
    fake_y = self.generator("G", x, reuse=False)
    return fake_y

  def eval(self):
    x = tf.placeholder(tf.float32, shape=[None, self.img_dim[0], self.img_dim[1], self.img_dim[2]], name='x')
    #
    y = self.inference(x)
    test_dir ="../data/apple2orange/testA"
    test_image=os.listdir(test_dir)
    #
    saver = tf.train.Saver()
    with tf.Session(config=CONFIG) as sess:
      ckpt = tf.train.get_checkpoint_state(self.model_path)
      saver.restore(sess, ckpt.model_checkpoint_path)
      print('model load done')
      input_img = np.zeros((1,256,256,3))
      for idx,img_path in enumerate(test_image):
        img_name =os.path.basename(img_path)
        img=imread(os.path.join(test_dir,img_path))
        img = (img / 127.5) - 1.0
        input_img[0] = img
        output_img = sess.run(y, feed_dict={x: input_img})
        # 数值从[-1, 1] -> [0,255]
        output_img = (255.0 * (output_img + 1.0) / 2.0)  # .astype('uint8')
        gen_img = output_img[0]
        # gen_img =gen_img[..., -1::-1]  # 如果是opencv则bgr->rgb
        imsave(os.path.join("../data/apple2orange/images",img_name), gen_img)



# ------------------------------------------------------------------------


def main():
  params =Params()
  model = GAN(params)
  if FLAGS.if_train ==True:
    # pass
    model.train()
  else:
    model.eval()


if __name__ == '__main__':
  main()