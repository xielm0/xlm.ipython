# -*- coding:utf-8 -*-
from datetime import datetime
import os
import time
import tensorflow as tf
import inference
import numpy as np
import input_data

# from tensorflow.python.ops import partitioned_variables
# from tensorflow.python.client import timeline

# 启动训练
# CLASSPATH=$(${HADOOP_HDFS_HOME}/bin/hadoop classpath --glob) python train.py

N=90
M=75  #多少个连续特征

BATCH_SIZE = 500
N_GPU = 4
BATCH_NUM = BATCH_SIZE * N_GPU
TRAINING_STEPS = 30000

LEARNING_RATE_BASE = 0.0003
LEARNING_RATE_DECAY = 0.99
MOVING_AVERAGE_DECAY = 0.99



# # 定义日志和模型输出的路径
TENSORBOARD_PATH ="./tensorboard_logs/"
MODEL_SAVE_PATH = "./models/"
MODEL_NAME = "model.ckpt"

CONFIG = tf.ConfigProto(
    allow_soft_placement=True,
    log_device_placement=False)
CONFIG.gpu_options.allow_growth = True


def get_loss(x, y_,  scope):
    y = inference.inference(x, train_flag=True )
    y = tf.reshape(y, (BATCH_SIZE,))

    # cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.cast(y_, tf.int32)))
    cross_entropy = -tf.reduce_mean(
        y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)) + (1 - y_) * tf.log(tf.clip_by_value(1 - y, 1e-10, 1.0)))
    # scope="GPU_i" ,so计算当前GPU上的loss
    regularization_loss = tf.add_n(tf.get_collection('losses', scope))
    loss = cross_entropy + regularization_loss

    return loss

# 同步模式，计算梯度的平均值
def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)

        v = grad_and_vars[0][1]
        grad_and_vars = (grad, v)
        average_grads.append(grad_and_vars)
    return average_grads


def main(argv=None):
    # clear tensorboard logs
    os.system("/bin/rm "+ TENSORBOARD_PATH + "*")
    # test data reading
    with tf.device('/cpu:0'):
        input_tensors,y_ = input_data.get_train_data(BATCH_NUM)

        global_step = tf.get_variable(
            'global_step', [], initializer=tf.constant_initializer(0),
            trainable=False)
        learning_rate = tf.train.exponential_decay(
            LEARNING_RATE_BASE, global_step, 100000 / BATCH_NUM,
            LEARNING_RATE_DECAY
        )
        tf.summary.scalar('learning-rate', learning_rate)

        # opt = tf.train.GradientDescentOptimizer(learning_rate)
        opt = tf.train.AdamOptimizer(learning_rate)
        # opt = tf.train.AdagradOptimizer(learning_rate)
        tower_grads = []

        # 记录每个GPU的损失函数值
        loss_gpu_dir = {}
        # 将神经网络的优化过程跑在不同的GPU上
        # 一个GPU使用其中 1/N_GPU 数据
        for i in xrange(N_GPU):
            with tf.device('/gpu:%d' % i):
                with tf.name_scope('GPU_%d' % i) as scope:
                    a = i* BATCH_SIZE
                    b = (i+1)* BATCH_SIZE
                    # input_tensors[2]是个list
                    tmp_onehot = map(lambda xi: xi[a:b,], input_tensors[2])
                    tmp_tensors=[input_tensors[0][a:b,],input_tensors[1][a:b,],tmp_onehot]
                    # 第一个GPU已经创建了varibales,其余GPU中使用模型参数时，需要reuse_variables
                    cur_loss = get_loss(tmp_tensors, y_[a:b,],  scope)
                    loss_gpu_dir['GPU_%d' % i] = cur_loss
                    #
                    tf.get_variable_scope().reuse_variables()
                    # 当前GPU计算当前梯度。
                    grads_and_vars  = opt.compute_gradients(cur_loss)
                    tower_grads.append(grads_and_vars )
        for los in loss_gpu_dir:
            tf.summary.scalar('GPU Loss/' + los, loss_gpu_dir[los])
        # 计算变量的平均梯度，并输出到TensorBoard日志
        grads = average_gradients(tower_grads)
        for grad, var in grads:
            if grad is not None:
                tf.summary.histogram( 'gradients_on_average/%s' % var.op.name, grad)
        # 使用平均梯度更新参数
        tf.get_variable_scope()._reuse = False
        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)

        # 计算变量的滑动平均值
        variable_averages = tf.train.ExponentialMovingAverage(
            MOVING_AVERAGE_DECAY, global_step)
        variables_averages_op = variable_averages.apply(
            tf.trainable_variables())
        tf.get_variable_scope()._reuse = True

        train_op = tf.group(apply_gradient_op, variables_averages_op)
        #
        saver = tf.train.Saver(tf.all_variables())
        summary_op = tf.summary.merge_all()

        init = tf.global_variables_initializer()

    with tf.Session(config=CONFIG) as sess:
        init.run()
        sess.run(tf.local_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        summary_writer = tf.summary.FileWriter(TENSORBOARD_PATH, sess.graph)

        for step in xrange(TRAINING_STEPS):

            if step != 0 and step % 100 == 0:
                # 性能分析
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                #
                start_time = time.time()
                # _, loss_value = sess.run([train_op, cur_loss])
                _, loss_value1, loss_value2, loss_value3, loss_value4 = sess.run([train_op] + loss_gpu_dir.values(),
                                                                                 options=run_options, run_metadata=run_metadata)
                duration = time.time() - start_time
                num_examples_per_step = BATCH_NUM
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = duration / N_GPU
                format_str = ('step %d, loss = %.10f, %.10f, %.10f, %.10f (%.1f examples sec; %.3f sec/batch)')
                print(format_str % (step, loss_value1, loss_value2, loss_value3, loss_value4, examples_per_sec, sec_per_batch))

                # 写tensorboard日志
                summary = sess.run(summary_op)
                summary_writer.add_run_metadata(run_metadata, 'step%03d' % step)
                summary_writer.add_summary(summary, step)
            else:
                start_time = time.time()
                # _, loss_value = sess.run([train_op, cur_loss])
                _, loss_value1, loss_value2, loss_value3, loss_value4 = sess.run([train_op] + loss_gpu_dir.values())
                duration = time.time() - start_time

            if step % 500 == 0 or (step + 1) == TRAINING_STEPS:
                checkpoint_path = os.path.join(MODEL_SAVE_PATH, MODEL_NAME)
                saver.save(sess, checkpoint_path, global_step=step)
        coord.request_stop()
        coord.join(threads)

if __name__ == '__main__':
    tf.app.run()

