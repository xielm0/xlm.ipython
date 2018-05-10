# -*- coding:utf-8 -*-
import tensorflow as tf
import logging
import input
import model
import time
import os,sys
# import numpy as np
# import math
# import six.moves.reduce as reduce

# graph params
param_dict={}
param_dict["hash_size"]=[100,1000,1000,10**5,10**5,10**5]
param_dict["embedding_size"]=[8,10,14,32,32,32]  # sum=128
param_dict["vocabulary_size"]=int(1e5)
param_dict["num_sampled"]=32


# train params
num_gpus=4
batch_size=1280*num_gpus
# train 1 billion examples
max_steps = (10**9)/batch_size
# multi_gpu or one_gpu
multi_gpu = 0
fineTune=True
timeLine=False
# tensorBoard switch
tensorBoard=False
tensorBoard_path ="../logs/tensorboard/"
# model
MODEL_SAVE_PATH = "../models/"
MODEL_NAME = "model.ckpt"

if not os.path.exists(tensorBoard_path):
    os.mkdir(tensorBoard_path)
if not os.path.exists(MODEL_SAVE_PATH):
    os.mkdir(MODEL_SAVE_PATH)


CONFIG = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
CONFIG.gpu_options.allow_growth = True


def init_learning_rate():
    #params
    inital_learning_rate=1e-3
    decay_steps=1000
    decayg_rate=0.99
    #
    global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0),trainable=False)
    learning_rate = tf.train.exponential_decay(inital_learning_rate, global_step, decay_steps, decayg_rate)
    tf.summary.scalar('learning-rate', learning_rate)
    #
    return global_step,learning_rate


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


def train_with_one_gpu(train_inputs,train_labels):
    global_step,learning_rate=init_learning_rate()
    opt = tf.train.GradientDescentOptimizer(learning_rate)
    sku2vec=model.sku2vec(param_dict)
    #
    i=1
    with tf.device('/gpu:%d' % i):
        loss = sku2vec.get_loss(train_inputs,train_labels)
        #train_op = opt.minimize(loss,global_step)
        grads_and_vars = opt.compute_gradients(loss)
        train_op = opt.apply_gradients(grads_and_vars,global_step=global_step)
    #
    for grad, var in grads_and_vars:
        tf.summary.scalar('gradients_on_average/%s' % var.op.name, tf.reduce_sum(tf.abs(grad)))
    return loss,train_op


def train_with_multi_gpu(train_inputs,train_labels,batch_size,num_gpus):
    global_step,learning_rate=init_learning_rate()
    opt = tf.train.GradientDescentOptimizer(learning_rate)
    sku2vec=model.sku2vec(param_dict)
    tower_grads = []
    # 记录每个GPU的损失函数值
    loss_gpu_dir = {}
    # 将神经网络的优化过程跑在不同的GPU上
    # 一个GPU使用其中 1/num_gpus 数据
    for i in range(num_gpus):
        with tf.device('/gpu:%d' % i):
            with tf.name_scope('GPU_%d' % i) as scope:
                a = batch_size *i/num_gpus
                b = batch_size *(i+1)/num_gpus
                train_inputs_i = train_inputs[a:b, ]
                train_labels_i = train_labels[a:b, ]
                #
                cur_loss = sku2vec.get_loss(train_inputs_i, train_labels_i )
                loss_gpu_dir['GPU_%d' % i] = cur_loss
                #第一个GPU已经创建了varibales,其余GPU中使用模型参数时，需要reuse_variables
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

    # 计算变量的滑动平均值
    # sku2vec这里可以先不考虑平滑
    # moving_average_decay=0.99
    # ema = tf.train.ExponentialMovingAverage(moving_average_decay, global_step)
    # variables_averages_op = ema.apply(tf.trainable_variables())
    # train_op = tf.group(apply_gradient_op, variables_averages_op)

    train_op = tf.group(apply_gradient_op)

    return loss_gpu_dir.get("GPU_0"), train_op


def train():
    #input
    with tf.device('/cpu:0' ):
        train_inputs, train_labels = input.get_train_batch(batch_size)

    if multi_gpu==0:
        loss,train_op =train_with_one_gpu(train_inputs,train_labels)
    else:
        loss,train_op =train_with_multi_gpu(train_inputs,train_labels,batch_size,num_gpus)

    if fineTune:
        saver = tf.train.Saver( tf.trainable_variables())
    else:
        saver = tf.train.Saver()

    if tensorBoard:
        # clear tensorboard logs
        cmd = "/bin/rm  %s/events.out.tfevents.*"%(tensorBoard_path)
        os.system(cmd)
        #
        for var in tf.trainable_variables():
            logging.info(var.op.name)
            tf.summary.scalar(var.op.name,tf.reduce_sum(tf.abs(var)))
            # tf.summary.histogram(var.op.name, var)
        merged_summary_op = tf.summary.merge_all()

    init = (tf.global_variables_initializer(), tf.local_variables_initializer())
    with tf.Session(config=CONFIG) as sess:
        sess.run(init)
        if fineTune:
            # restore variable
            # tf.train.get_checkpoint_state 会自动找到目录中最新模型的文件名。
            ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path :
                saver.restore(sess, ckpt.model_checkpoint_path)

        #
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord,sess=sess)
        #
        summary_writer = tf.summary.FileWriter(tensorBoard_path, sess.graph)
        if timeLine:
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()

        t1 = time.time()
        # 当 step > num_epochs*n/ batch_size ,则raise:
        try:
            for step in xrange(max_steps):
                if timeLine == False:
                    _, loss_vlaue = sess.run([train_op, loss] )
                elif timeLine and step >1000 and step < 1100:
                    _, loss_vlaue = sess.run([train_op, loss] , options=run_options, run_metadata=run_metadata)
                    from tensorflow.python.client import timeline
                    tl = timeline.Timeline(run_metadata.step_stats)
                    chrome_trace = tl.generate_chrome_trace_format()
                    with open('timeline_01.json', 'w') as f:
                        f.write(chrome_trace)

                if step != 0 and step % 100 == 0:
                    t2 = time.time()
                    duration =t2 -t1
                    t1=t2
                    num_examples = batch_size * 100
                    examples_per_sec = num_examples / duration
                    format_str = ('step %d, loss = %.10f (%.1f examples per sec)')
                    logging.info(format_str % (step, loss_vlaue, examples_per_sec))

                # write tensorBoard log
                if step != 0 and step % 100 == 0 and tensorBoard:
                    summary = sess.run(merged_summary_op)
                    summary_writer.add_summary(summary, step)

                # save graph ,include vars
                if step % 2000 == 0 or (step+1) == max_steps:
                    checkpoint_path = os.path.join(MODEL_SAVE_PATH,MODEL_NAME)
                    saver.save(sess, checkpoint_path, global_step=step)
        except tf.errors.OutOfRangeError:
            logging.info("train ended!")
        except Exception as e:
            logging.info("Exception type:%s"%type(e))
            logging.info("Unexpected Error: {}".format(e))
        finally:
            coord.request_stop()
            coord.join(threads)



def send_model(ip):
    #delete
    # delte_cmd='find %s -type f -name "model.ckpt*" -mtime +3 -exec rm  {} \;' % MODEL_SAVE_PATH
    username = "admin"
    local_path =MODEL_SAVE_PATH
    remote_path ="/export/biz/sku2vec/models/"
    # sync_model_cmd = './.sshpass -p "%s" scp -r %s %s@%s:%s' % (password, local_path,  username, ip, remote_path)
    sync_model_cmd = 'scp -r %s %s@%s:%s' % ( local_path,  username, ip, remote_path)
    logging.info(sync_model_cmd)
    os.system(sync_model_cmd)

def main():
    logging.basicConfig(level=logging.INFO)
    #
    input.init_env()
    train()
    # send model
    import download
    cur_ip = download.get_ip()
    for ip in download.worker_machine:
        if cur_ip  != ip:
            send_model(ip)


if __name__ == '__main__':
    main()

