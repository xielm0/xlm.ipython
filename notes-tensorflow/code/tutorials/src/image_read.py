# -*- coding:utf-8 -*-
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import numpy as np

CONFIG = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
CONFIG.gpu_options.allow_growth = True

jpg_path = "../data/image_exame/1.jpg"

def plt_test():
    # read , rgb顺序
    img = plt.imread(jpg_path)
    print(type(img),img.dtype) #<class 'numpy.ndarray'> uint8
    # save
    plt.imsave("../data/tmp/test1.jpg",img)
    # convert
    # resize
    print(img.shape) #450, 450, 3
    img.reshape(450*450*3)

    # show
    plt.imshow(img)
    plt.show() # 才会真正show

def cv2_test():
    # read , bgr顺序
    img = cv2.imread(jpg_path)
    print(type(img),img.dtype)
    # save
    cv2.imwrite("../data/tmp/test2.jpg",img)
    # convert
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # BGR转灰度
    # img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  # 灰度转BRG
    # img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # 也可以灰度转RGB
    img = img[..., -1::-1] #bgr -> rgb
    # resize
    img = cv2.resize(img,(128,128))
    # show
    cv2.imshow("test.jpg",img)
    cv2.waitKey(0)
    # plt show
    plt.imshow(img) # 因为opencv读取进来的是bgr顺序呢的，而imshow需要的是rgb顺序，因此需要先反过来
    plt.show()


def pil_test():
    # read
    img = Image.open(jpg_path)
    print(type(img) )
    # save
    img.save("../data/tmp/test3.jpg" )
    # convert
    # img = img.convert('L')  # RGB转换成灰度图像
    # img = img.convert('RGB')  # 灰度转RGB
    # rgb->bgr
    r, g, b = img.split()
    img = Image.merge('RGB', (b, g, r))
    img = img.resize((128,128))
    # show
    Image._show(img)
    # plt show
    img = np.array(img)  # 转成ndarray
    plt.imshow(img)
    plt.show() # 才会真正show


def tf_test():
    # 图片编码
    # image_raw = tf.read_file(jpg_path)  # Tensor(, dtype=string)
    with tf.gfile.FastGFile(jpg_path, 'rb') as f:
        image_raw=f.read()  # string，内容等同image_raw.eval()
    print(image_raw)
    #
    # 图片解码
    image = tf.image.decode_jpeg(image_raw)
    # image = tf.image.resize_images(image, size=(128,128))
    with tf.Session() as sess:
        img = sess.run(image)
    # show
    plt.imshow(img)
    plt.show()


def tf_test2():
    #
    image_raw = plt.imread(jpg_path)
    image_raw = image_raw.tobytes()
    #
    image = tf.decode_raw(image_raw, tf.uint8)
    image = tf.reshape(image, [450, 450, 3])
    with tf.Session() as sess:
        img = sess.run(image)
    # show
    plt.imshow(img)
    plt.show()

def imread():
    img = plt.imread(jpg_path)
    img = img.tobytes()

def main():
    # plt_test()
    # cv2_test()
    # pil_test()
    tf_test2()


if __name__ == "__main__":
    main()
