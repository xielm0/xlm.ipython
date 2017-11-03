import numpy as np
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import tensorflow.contrib.slim as slim


STEPS=10

def generate_data(seq):
    x=[]
    y=[]
    for i in range(len(seq)-STEPS-1):
        x.append([seq[i:i+STEPS]])
        x.append([seq[i+STEPS]])
    return np.array(x),np.array(y)

def main():
    net=slim.conv2d(input,num_outputs=32,kernel_size=[3,3])
    pass

