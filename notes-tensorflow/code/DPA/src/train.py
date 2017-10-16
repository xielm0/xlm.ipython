# -*- coding:utf-8 -*-
import os
import time
import tensorflow as tf

import inference
import input


# 启动训练
# CLASSPATH=$(${HADOOP_HDFS_HOME}/bin/hadoop classpath --glob) python train.py

N=input.N
M=input.M  #多少个连续特征

BATCH_SIZE = 500
N_GPU = 4
BATCH_NUM = BATCH_SIZE * N_GPU
TRAINING_STEPS = 20000

LEARNING_RATE = 0.01
DECAYG_RATE = 0.97
DECAY_STEPS = 100
MOVING_AVERAGE_DECAY = 0.99


# 定义日志和模型输出的路径
TENSORBOARD_PATH ="../logs/tensorboard/"
MODEL_SAVE_PATH = "../models/"
MODEL_NAME = "model.ckpt"

if not os.path.exists(TENSORBOARD_PATH):
    os.mkdir(TENSORBOARD_PATH)
if not os.path.exists(MODEL_SAVE_PATH):
    os.mkdir(MODEL_SAVE_PATH)

CONFIG = tf.ConfigProto(
    allow_soft_placement=True,
    log_device_placement=False)
CONFIG.gpu_options.allow_growth = True


# 备份tensorboard日志
def backup_tensorboardLogs():
    # cmd="find %s -type d   -mtime +1  -exec rm -R {} \;" %(TENSORBOARD_PATH)
    os.system(cmd)
    match = 'events.out.tfevents.'
    for file_name in os.listdir(TENSORBOARD_PATH):
        if match in file_name:
            dir_name= file_name.split(match)[1]
            dir_name_path= TENSORBOARD_PATH + dir_name
            if not os.path.exists(dir_name_path):
                os.mkdir(dir_name_path)
            cmd="/bin/mv " + TENSORBOARD_PATH + file_name+"  "+dir_name_path
            print(cmd)
            print("running .....")
            os.system(cmd)



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


def build_model(x_list, y_):
    """

    :param x_list:
    :param y_:
    :return: global_step, loss =loss_gpu_dir, train_op, opt
    """
    global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0),
                                  trainable=False)
    learning_rate = tf.train.exponential_decay(LEARNING_RATE, global_step, DECAY_STEPS,DECAYG_RATE)
    tf.summary.scalar('learning-rate', learning_rate)

    # opt = tf.train.GradientDescentOptimizer(learning_rate)
    opt = tf.train.AdamOptimizer(learning_rate)
    # opt = tf.train.AdagradOptimizer(learning_rate)

    tower_grads = []
    # 记录每个GPU的损失函数值
    loss_gpu_dir = {}
    # 将神经网络的优化过程跑在不同的GPU上
    # 一个GPU使用其中 1/N_GPU 数据
    for i in range(N_GPU):
        with tf.device('/gpu:%d' % i):
            with tf.name_scope('GPU_%d' % i) as scope:
                a = i* BATCH_SIZE
                b = (i+1)* BATCH_SIZE
                # x_list[2]是个list
                x_category_max =x_list[2]
                tmp_tensors=[x_list[0][a:b, ],x_list[1][a:b, ],x_category_max]
                # 第一个GPU已经创建了varibales,其余GPU中使用模型参数时，需要reuse_variables
                cur_loss = inference.get_loss(tmp_tensors, y_[a:b, ],  scope)
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
    ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = ema.apply(tf.trainable_variables())

    train_op = tf.group(apply_gradient_op, variables_averages_op)

    return global_step, loss_gpu_dir, train_op, opt


def main(argv=None):
    # clear tensorboard logs
    # cmd = "/bin/rm  " + TENSORBOARD_PATH + "events.out.tfevents.*"
    # os.system(cmd)
    backup_tensorboardLogs()
    # test data reading
    with tf.device('/cpu:0'):
        with tf.name_scope("input"):
            x_list, y_ = input.get_train_data(BATCH_NUM)

        global_step, loss_gpu_dir, train_op, opt = build_model(x_list, y_)
        #
        saver = tf.train.Saver(tf.all_variables())
        summary_op = tf.summary.merge_all()

        init = tf.global_variables_initializer()

    with tf.Session(config=CONFIG) as sess:
        init.run()
        sess.run(tf.local_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord,sess=sess)
        summary_writer = tf.summary.FileWriter(TENSORBOARD_PATH, sess.graph)

        for step in xrange(TRAINING_STEPS):

            if step != 0 and step % 100 == 0:
                # 性能分析
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                #
                t1 = time.time()
                # _, loss_value = sess.run([train_op, cur_loss])
                _, loss_value1, loss_value2, loss_value3, loss_value4 = sess.run([train_op] + loss_gpu_dir.values(),
                                                                                 options=run_options, run_metadata=run_metadata)
                duration = time.time() - t1
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
                # 仅仅执行
                _, loss_value1, loss_value2, loss_value3, loss_value4 = sess.run([train_op] + loss_gpu_dir.values())

            if step % 1000 == 0 or (step + 1) == TRAINING_STEPS:
                checkpoint_path = os.path.join(MODEL_SAVE_PATH, MODEL_NAME)
                saver.save(sess, checkpoint_path, global_step=step)

        coord.request_stop()
        coord.join(threads)

if __name__ == '__main__':
    tf.app.run()

