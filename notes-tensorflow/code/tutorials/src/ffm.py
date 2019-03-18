# -*- coding: UTF-8 -*-

import tensorflow as tf
import pandas as pd
import numpy as np
import os
input_x_size = 80;
field_size = 8;
vector_dimension = 3;
total_plan_train_steps = 1000;
MODEL_SAVE_PATH = "TFModel"
MODEL_NAME = "FFM"
BATCH_SIZE = 1;



def createOneDimensionWeight(input_x_size):
    weights = tf.truncated_normal([input_x_size])
    tf_weights = tf.Variable(weights)
    return tf_weights;

def createTwoDimensionWeight(input_x_size,
                   field_size,
                   vector_dimension):
    weights = tf.truncated_normal([int(input_x_size * (input_x_size + 1) / 2),
                                   field_size,
                                   vector_dimension
                                   ])
    tf_weights = tf.Variable(weights);
    return tf_weights;

def inference(input_x, input_x_field):
    """计算回归模型输出的值"""
    zeroWeights = createZeroDimensionWeight();  # 随机初始化常数项的权重
    oneDimWeights = createOneDimensionWeight(input_x_size);  # 随机初始化一次项的权重

    secondValue = tf.reduce_sum(tf.multiply(oneDimWeights, input_x, name="secondVale"));  # 计算一次项的权重和x的点积，和点积后的和
    firstTwoValue = tf.add(zeroWeights, secondValue, name="firstTwoValue");  # 常数项和一次项的值

    thirdWeight = createTwoDimensionWeight(input_x_size,  # 创建二次项的权重变量
                                           field_size,
                                           vector_dimension);

    thirdValue = tf.Variable(0.0, dtype=tf.float32);  # 初始化二次项的和为0
    input_shape = input_x_size;  # 得到输入训练数据的大小
    for i in range(input_shape):
        featureIndex1 = i;  # 第一个特征的索引编号
        fieldIndex1 = int(input_x_field[i]);  # 第一个特征所在域的索引编号
        for j in range(i + 1, input_shape):
            featureIndex2 = j;  # 第二个特征的索引编号
            fieldIndex2 = int(input_x_field[j]);  # 第二个特征的所在域索引编号
            vectorLeft = tf.convert_to_tensor(
                [[featureIndex1, fieldIndex2, 0], [featureIndex1, fieldIndex2, 1], [featureIndex1, fieldIndex2, 2]])
            weightLeft = tf.gather_nd(thirdWeight, vectorLeft)
            weightLeftAfterCut = tf.squeeze(weightLeft)

            vectorRight = tf.convert_to_tensor(
                [[featureIndex2, fieldIndex1, 0], [featureIndex2, fieldIndex1, 1], [featureIndex2, fieldIndex1, 2]])
            weightRight = tf.gather_nd(thirdWeight, vectorRight)
            weightRightAfterCut = tf.squeeze(weightRight)
            tempValue = tf.reduce_sum(tf.multiply(weightLeftAfterCut, weightRightAfterCut))

            indices2 = [i]
            indices3 = [j]

            xi = tf.squeeze(tf.gather_nd(input_x, indices2));
            xj = tf.squeeze(tf.gather_nd(input_x, indices3));

            product = tf.reduce_sum(tf.multiply(xi, xj));

            secondItemVal = tf.multiply(tempValue, product)

            tf.assign(thirdValue, tf.add(thirdValue, secondItemVal))

    fowardY = tf.add(firstTwoValue, thirdValue)

    return fowardY;


if __name__ == "__main__":
    global_step = tf.Variable(0, trainable=False)
    (train_x, train_y, train_x_field) = read_csv();
    input_x = tf.placeholder(tf.float32, [None, 80])
    input_y = tf.placeholder(tf.float32, [None, 1])
    y_ = inference(input_x, train_x_field)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=y_, labels=input_y);
    train_step = tf.train.GradientDescentOptimizer(0.001, name="GradientDescentOptimizer").minimize(cross_entropy,
                                                                                                    global_step=global_step);

    saver = tf.train.Saver();
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in range(total_plan_train_steps):
            input_x_batch = train_x[int(i * BATCH_SIZE):int((i + 1) * BATCH_SIZE)]
            input_y_batch = train_y[int(i * BATCH_SIZE):int((i + 1) * BATCH_SIZE)]

            predict_loss, steps = sess.run([train_step, global_step],
                                           feed_dict={input_x: input_x_batch, input_y: input_y_batch})
            if (i + 1) % 2 == 0:
                print("After  {step} training   step(s)   ,   loss    on    training    batch   is  {predict_loss} "
                      .format(step=steps, predict_loss=predict_loss))

                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=steps)
                writer = tf.summary.FileWriter(os.path.join(MODEL_SAVE_PATH, MODEL_NAME), tf.get_default_graph())
                writer.close()