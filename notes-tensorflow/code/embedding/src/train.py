# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
import input
import math
import time
import os
# import six.moves.reduce as reduce
from multiprocessing import Process
from log import logger

N_GPU=4
learning_rate=0.001
batch_size =128
num_sampled=64
vocabulary_size=100000+1

discrete_nums= 5
embedding_size_list=[8,12,16,32,32]
embedding_size=sum(embedding_size_list)


TENSORBOARD_PATH ="../logs/tensorboard/"
MODEL_SAVE_PATH = "../models/"
MODEL_NAME = "model.ckpt"

cid_list=input.cid_list
run_cid_list=input.run_cid_list

for cid in cid_list:
    path = os.path.join(MODEL_SAVE_PATH,cid)
    if not os.path.exists(path):
        os.mkdir(path)
    path = os.path.join(TENSORBOARD_PATH,cid)
    if not os.path.exists(path):
        os.mkdir(path)

CONFIG = tf.ConfigProto(
    allow_soft_placement=True,
    log_device_placement=False)
CONFIG.gpu_options.allow_growth = True



def average_gradients(tower_grads):
    """
    同步模式，计算梯度的平均值
    :param tower_grads: 一个list ,里面元素是4个list , 每个list的元素是turple类似：a=[[(1,2),(2,3)],[(101,2),(102,3)]]
    zip(*a), 即[((1, 2), (101, 2)), ((2, 3), (102, 3))]
    :return:
    """
    with tf.name_scope("average_grads"):
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            grads = []
            for g, _ in grad_and_vars:
                expanded_g = tf.expand_dims(g, 0)
                grads.append(expanded_g)
            grad = tf.concat(grads, 0)
            grad = tf.reduce_mean(grad, 0)

            var = grad_and_vars[0][1]
            grad_and_vars = (grad, var)
            average_grads.append(grad_and_vars)
        return average_grads


def build_graph(train_inputs, train_labels, cid):
    # data processing,对离散的数据进行onehot
    x_max  = input.get_max(cid)

    #
    embed_list = []
    for i in range(discrete_nums):
        with tf.variable_scope('embedding_%d' % i):
            m = x_max[i]+1
            EmbedWeights = tf.get_variable("EmbedWeights", shape=[m, embedding_size_list[i]],
                                           initializer=tf.random_uniform_initializer(-1.0, 1.0))
            embed_i = tf.nn.embedding_lookup(EmbedWeights, tf.cast(train_inputs[:, i], tf.int32))
            embed_list.append(embed_i)
    embed = reduce(lambda a, b: tf.concat([a, b], 1), embed_list)

    # Construct the variables for the NCE loss
    nce_weights = tf.Variable(
        tf.truncated_normal([vocabulary_size, embedding_size], stddev=1.0 / math.sqrt(embedding_size)))
    nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

    #  logits = tf.matmul(inputs, tf.transpose(weights)) + biases
    loss = tf.reduce_mean(
        tf.nn.nce_loss(weights=nce_weights,
                       biases=nce_biases,
                       labels=train_labels,
                       inputs=embed,
                       num_sampled=num_sampled,
                       num_classes=vocabulary_size))

    with tf.name_scope("loss"):
        tf.summary.scalar("loss", loss)

    return loss

"""
 这里有2种方案：
 1，4个gpu跑一个cid的任务
 2，4个gpu分别跑一个cid的任务，且并行。
 分析：
 第1种方案，只是由于batch_size是1/4，从而计算速度快。但每个gpu都建立了一个graph,而这个graph只有1/4
 第2种方案，更适合独立的任务运行。
"""
def build_model_single(train_inputs,train_labels,cid):
    inital_learning_rate=1e-3
    decay_steps=1000
    decayg_rate = 0.99

    global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0),trainable=False)
    learning_rate = tf.train.exponential_decay(inital_learning_rate, global_step, decay_steps, decayg_rate)
    tf.summary.scalar('learning-rate', learning_rate)

    opt = tf.train.GradientDescentOptimizer(learning_rate)
    i= run_cid_list.index(cid) % N_GPU
    with tf.device('/gpu:%d' % i):
        loss = build_graph(train_inputs,train_labels,cid)
        train_op = opt.minimize(loss,global_step)
    return loss,train_op


def build_model_multi(train_inputs,train_labels,cid):
    inital_learning_rate=1e-3
    decay_steps=1000
    decayg_rate = 0.99

    global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0),trainable=False)
    learning_rate = tf.train.exponential_decay(inital_learning_rate, global_step, decay_steps, decayg_rate)
    tf.summary.scalar('learning-rate', learning_rate)

    opt = tf.train.GradientDescentOptimizer(learning_rate)

    tower_grads = []
    # 记录每个GPU的损失函数值
    loss_gpu_dir = {}
    # 将神经网络的优化过程跑在不同的GPU上
    # 一个GPU使用其中 1/N_GPU 数据
    for i in range(N_GPU):
        with tf.device('/gpu:%d' % i):
            with tf.name_scope('GPU_%d' % i) as scope:
                a = i* batch_size/N_GPU
                b = (i+1)* batch_size/N_GPU
                # 第一个GPU已经创建了varibales,其余GPU中使用模型参数时，需要reuse_variables
                cur_loss = build_graph(train_inputs, train_labels,cid)
                loss_gpu_dir['GPU_%d' % i] = cur_loss
                #
                tf.get_variable_scope()._reuse = True
                # 当前GPU计算当前梯度。
                # Returns:  A list of (gradient, variable) pairs
                grads_and_vars = opt.compute_gradients(cur_loss)
                tower_grads.append(grads_and_vars)

    # 计算变量的平均梯度
    grads = average_gradients(tower_grads)

    for grad, var in grads:
        if grad is not None:
            tf.summary.scalar('gradients_on_average/%s' % var.op.name, tf.reduce_sum(tf.abs(grad)))
            #tf.summary.histogram('gradients_on_average/%s' % var.op.name, grad)

    # 使用平均梯度更新参数
    tf.get_variable_scope()._reuse = False
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)

    # 计算变量的滑动平均值
    moving_average_decay=0.99
    ema = tf.train.ExponentialMovingAverage(moving_average_decay, global_step)
    variables_averages_op = ema.apply(tf.trainable_variables())

    train_op = tf.group(apply_gradient_op, variables_averages_op)

    return loss_gpu_dir, train_op

def train(cid):
    # clear tensorboard logs
    tensorboard_path=TENSORBOARD_PATH+cid
    cmd = "/bin/rm  %s/events.out.tfevents.*"%(tensorboard_path)
    os.system(cmd)

    #input
    i= run_cid_list.index(cid) % 2
    with tf.device('/cpu:%d' %i ):
        batch_inputs, batch_labels = input.read_TFRecords(batch_size,cid)

    # train_inputs = tf.placeholder(tf.int32, shape=[batch_size,discrete_nums])
    # train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    (train_inputs,train_labels) = (batch_inputs,batch_labels)

    #
    loss,train_op =build_model_single(train_inputs,train_labels,cid)

    saver = tf.train.Saver()
    merged_summary_op = tf.summary.merge_all()


    with tf.Session(config=CONFIG) as sess:
        # We must initialize all variables before we use them.
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()
        print("Initialized")
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord,sess=sess)
        summary_writer = tf.summary.FileWriter(tensorboard_path, sess.graph)

        num_steps=200001
        for step in range(num_steps):
            # batch_x,batch_y = sess.run([batch_inputs, batch_labels])

            t1 = time.time()
            # _, loss_vlaue = sess.run([train_op, loss], feed_dict={train_inputs : batch_x, train_labels : batch_y})
            _, loss_vlaue = sess.run([train_op, loss] )
            t2 = time.time()
            duration =t2 -t1

            if step != 0 and step % 100 == 0:
                num_examples_per_step = batch_size
                examples_per_sec = num_examples_per_step / duration
                format_str = ('step %d, loss = %.10f (%.1f examples sec)')
                logger.info(format_str % (step, loss_vlaue, examples_per_sec))

                # 写tensorboard日志
                summary = sess.run(merged_summary_op)
                summary_writer.add_summary(summary, step)

            # save
            if step % 2000 == 0 or (step + 1) == num_steps:
                checkpoint_path = os.path.join(MODEL_SAVE_PATH,cid, MODEL_NAME)
                saver.save(sess, checkpoint_path, global_step=step)

        coord.request_stop()
        coord.join(threads)

def main():
    # train(cid)
    # 多进程：
    for idx,cid in enumerate(run_cid_list):
        logger.info("start train cid=" + cid)
        p=Process(target=train,args=(cid,))
        p.start()
        p.join()


if __name__ == '__main__':
    main()

