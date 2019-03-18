# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os
import time


image_path = "../data/image_exame/"  # 路径

if not os.path.exists(image_path):
    os.mkdir(image_path)


def phash(image,shape=(32,32),dct_flag=True,name=None):
    image = cv2.resize(image, shape)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    # gray=mvn(gray)
    if dct_flag:
        gray=cv2.dct(np.float32(gray))
        gray =gray[0:8, 0:8]
    # cv2.imwrite(os.path.join("../data/tmp", name), gray)
    avreage1 = np.mean(gray)
    shape=gray.shape
    image_hash = []
    for i in range(shape[0]):
        for j in range(shape[1]):
            if gray[i, j] > avreage1:
                image_hash.append(1)
            else:
                image_hash.append(0)
    return image_hash


def compute_sim(hash1,hash2):
    # 计算汉明距离
    num = 0
    for i in range(len(hash1)):
        if hash1[i] != hash2[i]:
            num += 1
    sim = 1 - num / len(hash1)
    return sim


def mvn(img):
    mean, std = cv2.meanStdDev(img)
    mean,std =mean[0][0],std[0][0]
    img =  (img-mean)/std/2
    img = (255.0*(img+1)/2)
    # cv2.imwrite("../data/tmp/test.jpg",img)
    return img


def main():
    image_list = os.listdir(image_path)
    image_list = filter(lambda x: x.endswith(".jpg"),image_list)
    name1 = "26.jpg"
    t1=time.time()
    img1 = cv2.imread(os.path.join(image_path,name1))
    print("img1=",img1)
    hash1 =phash(img1,shape=(32,32),dct_flag=True,name=name1)
    for name2 in image_list:
        # if name2!="12.jpg":
        #     continue
        img2=cv2.imread(os.path.join(image_path,name2))
        hash2 = phash(img2,shape=(32,32), dct_flag=True, name=name2) # 每个图片的phash值可以保存到内存数据库,加快速度
        sim=compute_sim(hash1,hash2)
        print(name1,name2,sim)
    t2=time.time()
    print(t2-t1)

if __name__ == '__main__':
    main()



