# -*- coding: utf-8 -*-
import time
import tensorflow as tf
from sklearn import metrics
import inference
import train
import numpy as np
import input_data
import processing as pr

N=90
M=75  #多少个连续特征

# CLASSPATH=$(${HADOOP_HDFS_HOME}/bin/hadoop classpath --glob) python eval.py
TEST_BATCH_SIZE = 10000
TEST_STEPS = 20
EVAL_INTERVAL_SECS = 10

CONFIG = tf.ConfigProto(
    allow_soft_placement=True,
    log_device_placement=False)
CONFIG.gpu_options.allow_growth = True

def evaluate():
    with tf.Graph().as_default() as g:
        # 获取测试数据
        # x = tf.placeholder(tf.float32, [None, 91], name='x-input')
        # y_ = tf.placeholder(tf.int64, [None, ], name='y-input')
        x, y_ = input_data.get_eval_data(TEST_BATCH_SIZE)
        y_= tf.reshape(y_, (-1,))
        x_continuous = x[:, :M]
        x_category = x[:, M:]
        x_category_onehot= pr.oneHot(x_category)
        input_tensors = [x_continuous, x_category, x_category_onehot]


        # 前向传播
        y = inference.inference(input_tensors, False )

        #
        saver = tf.train.Saver()

        while True:
            with tf.Session(config=CONFIG) as sess:
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(coord=coord)

                # tf.train.get_checkpoint_state 会自动找到目录中最新模型的文件名。
                ckpt = tf.train.get_checkpoint_state(train.MODEL_SAVE_PATH)

                if ckpt and ckpt.model_checkpoint_path:
                    # 加载模型
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    # 通过文件名获得模型保存时的迭代的轮数
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]

                    # 评估，这里是AUC评估。使用sk-learn的metrics
                    y_test = []
                    y_pred = []
                    for i in xrange(TEST_STEPS):
                        # print('loading test data...')
                        y_t, y_p = sess.run([y_, y])
                        y_test += y_t.tolist()
                        y_pred += y_p.flatten().tolist()
                    y_test = np.array(y_test)
                    y_pred = np.array(y_pred)
                    print(y_pred.shape)

                    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
                    auc_scroe = metrics.auc(fpr, tpr)
                    print("After %s training step(s), AUC score = %g" % (global_step, auc_scroe))
                else:
                    print('No checkpoint file found')
                coord.request_stop()
                coord.join(threads)
            time.sleep(EVAL_INTERVAL_SECS)

def main(argv=None):
    evaluate()

if __name__ == '__main__':
    tf.app.run()



