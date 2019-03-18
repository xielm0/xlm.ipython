# -*- coding:utf-8 -*-
import os
import time

import tensorflow as tf
import mnist_inference_cnn as inference
from tensorflow.examples.tutorials.mnist import input_data

BATCH_SIZE = 128
LEARNING_RATE_BASE = 1e-4
LEARNING_RATE_DECAY = 0.99
REGULARAZTION_RATE = 1e-4
MAX_TRAINING_STEPS = 10000
MOVING_AVERAGE_DECAY = 0.99
N_GPU = 4

# 定义路径
TENSORBOARD_PATH ="../logs/tensorboard/"
MODEL_SAVE_PATH = "../models/"
MODEL_NAME = "mnist.ckpt"

SESS_CONFIG=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
SESS_CONFIG.gpu_options.allow_growth = True

def get_loss(x, y_, scope):
    y = inference.inference(x, True)
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
    # regularization_loss = tf.add_n(tf.get_collection('losses', scope))
    regularization_loss=0
    loss = cross_entropy + regularization_loss
    with tf.name_scope("loss"):
        tf.summary.scalar('loss',loss)

    return loss

# 计算每一个变量的平均梯度
def average_gradients(tower_grads):
    """
    同步模式，计算梯度的平均值
    :param tower_grads: 一个list ,里面元素是4个list , 每个list的元素是turple类似：a=[[(1,2),(2,3)],[(101,2),(102,3)]]
    zip(*a), 即[((1, 2), (101, 2)), ((2, 3), (102, 3))]
    :return:
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            # expand_dims(input, axis=None, name=None, dim=None)
            # shape上增加一维，新增加的维度是axis维。
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)
        # concat,拼接, axis=0,按axis=0的维度拼接，也就是按行拼接。axis=0的维度数为增加。比如2个2行的矩阵，拼接后就是4行。
        grad = tf.concat(grads, axis=0)
        grad = tf.reduce_mean(grad, 0)

        v = grad_and_vars[0][1]
        grad_and_vars = (grad, v)
        average_grads.append(grad_and_vars)
    return average_grads


def build_graph(x, y_):
    global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
    learning_rate = LEARNING_RATE_BASE

    # opt = tf.train.GradientDescentOptimizer(learning_rate)
    opt = tf.train.AdamOptimizer(learning_rate)

    tower_grads = []
    # 记录每个GPU的损失函数值
    loss_gpu_dir = {}

    # 将神经网络的优化过程跑在不同的GPU上
    for i in range(N_GPU):
        with tf.device('/gpu:%d' % i):
            with tf.name_scope('GPU_%d' % i) as scope:
                cur_loss = get_loss(x, y_, scope)
                loss_gpu_dir['GPU_%d' % i] = cur_loss
                # 为了不同gpu更新同一组组参数，需要将
                tf.get_variable_scope().reuse_variables()
                # 当前GPU计算当前梯度。
                grads = opt.compute_gradients(cur_loss)
                tower_grads.append(grads)

    # 计算变量的平均梯度
    grads_and_var = average_gradients(tower_grads)
    for grad, var in grads_and_var:
        if grad is not None:
            tf.summary.histogram( 'gradients_on_average/%s' % var.op.name, grad)

    # 使用平均梯度更新参数
    tf.get_variable_scope()._reuse = False
    apply_gradient_op = opt.apply_gradients(grads_and_var, global_step=global_step)
    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)

    # 计算变量的滑动平均值
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    train_op = tf.group(apply_gradient_op, variables_averages_op)

    return global_step, loss_gpu_dir, train_op, opt



def main(argv=None):
    # 将简单运算放在CPU上,只有神经网络的训练放在GPU上。
    with tf.device('/cpu:0'):
        # 获取一个batch的数据,x [100, 784]
        x = tf.placeholder(tf.float32, [None, 784])
        y_ = tf.placeholder(tf.float32, [None, 10])
        #
        mnist = input_data.read_data_sets("/export/Data/MNIST_data", dtype=tf.uint8, one_hot=True)

    # build graph
    global_step, loss_gpu_dir, train_op, opt = build_graph(x, y_)

    # tensorboard
    cmd = "/bin/rm  " + TENSORBOARD_PATH + "events.out.tfevents.*"
    os.system(cmd)
    summary_op = tf.summary.merge_all()
    #
    all_vars = tf.trainable_variables()
    for v in all_vars:
      print(v)

    #
    saver = tf.train.Saver()
    init = tf.global_variables_initializer()
    with tf.Session(config=SESS_CONFIG) as sess:
        sess.run(init)
        #
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        #
        summary_writer = tf.summary.FileWriter(TENSORBOARD_PATH, sess.graph)

        for step in range(MAX_TRAINING_STEPS):
            batch = mnist.train.next_batch(128)
            start_time = time.time()
            _, loss_value1, loss_value2, loss_value3, loss_value4 = sess.run([train_op] + loss_gpu_dir.values(),
                                                                             feed_dict={x: batch[0], y_: batch[1]} )
            duration = time.time() - start_time

            if step != 0 and step % 100 == 0:
                num_examples_per_step = BATCH_SIZE * N_GPU
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = duration / N_GPU

                format_str = ('step %d, loss = %.10f, %.10f, %.10f, %.10f (%.1f examples sec; %.3f sec/batch)')
                print(format_str % (step, loss_value1, loss_value2, loss_value3, loss_value4, examples_per_sec, sec_per_batch))

                # 写tensorboard日志
                summary = sess.run(summary_op, feed_dict={x: batch[0], y_: batch[1]})
                summary_writer.add_summary(summary, step)

            if step % 1000 == 0 or (step + 1) == MAX_TRAINING_STEPS:
                checkpoint_path = os.path.join(MODEL_SAVE_PATH, MODEL_NAME)
                saver.save(sess, checkpoint_path, global_step=step)

        coord.request_stop()
        coord.join(threads)

if __name__ == '__main__':
    tf.app.run()

