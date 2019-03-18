# -*- coding:utf-8 -*-
import tensorflow as tf
import sys
import os

print(sys.argv[0])  #脚本的名称
# print(sys.argv[1])

CONFIG = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
CONFIG.gpu_options.allow_growth = True


from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2


def imread( img_file):
    img = Image.open(img_file)
    r, g, b = img.split()
    img = Image.merge('RGB', (b, g, r))
    img=np.array(img)
    return img

def read_image():
    a='../data/CycleGAN/apple2orange/trainA/n07740461_2.jpg'
    print(os.path.basename(a))

    b=[1,2,3]
    print(len(b))

    img1 = imread('../data/CycleGAN/apple2orange/trainA/n07740461_2.jpg')
    # print("img1=",img1)

    img2 =cv2.imread('../data/CycleGAN/apple2orange/trainA/n07740461_2.jpg')
    # print("img2=",img2)
    tf.layers.batch_normalization()


def test_read():
    batch_size=2
    epochs=2
    TFrecord_name="../data/apple2orange/trainA.tfrecords"
    filename_queue = tf.train.string_input_producer([TFrecord_name], num_epochs=epochs)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read_up_to(filename_queue,100)

    features = tf.parse_example(
        serialized_example,
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
        })
    # 将string转换为数字
    decoded_image = tf.cast(tf.decode_raw(features['image_raw'], tf.uint8), tf.float32)
    image = tf.reshape(decoded_image, [-1,256, 256, 3])
    print("image.shape=", image.shape.as_list())

    min_after_dequeue = 100
    capacity = min_after_dequeue + 3 * batch_size
    x = tf.train.shuffle_batch(
        [image],
        batch_size=batch_size,
        capacity=capacity,
        min_after_dequeue=min_after_dequeue, enqueue_many=True )

    print("x.shape=", x.shape.as_list())
    with tf.Session(config=CONFIG) as sess:
      init = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
      sess.run(init)
      #
      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(coord=coord, sess=sess)
      # print(decoded_image.eval())
      print(sess.run(x))


def test_image():
    # 图片编码
    # image_raw = tf.gfile.FastGFile("../data/apple2orange/trainA/n07740461_20.jpg", 'rb').read()
    # image_raw = tf.read_file('../data/apple2orange/trainA/n07740461_20.jpg')
    # image_raw = cv2.imread("../data/apple2orange/trainA/n07740461_20.jpg")
    # # 图片解码
    # image = tf.image.decode_jpeg(image_raw)
    # with tf.Session(config=CONFIG) as sess:
    #     img = sess.run(image)

    img = cv2.imread("../data/apple2orange/trainA/n07740461_106.jpg")
    # cv2.imshow("test.jpg", img)
    # cv2.waitKey(0)
    plt.imshow(img)
    plt.show()


def distcp():
    pass
def main2():
    # test()
    gama =tf.random_uniform([1,1,1,3])
    mean = tf.ones([2, 1, 1, 3])
    r =gama * mean
    # print(r.shape.as_list())
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        a,b=sess.run([gama,r])
        print("gamma=",a)
        print("res=",b)


def smooth():
    x = [10,9,8,7,6,5]

if __name__ == "__main__":
    test_image()
