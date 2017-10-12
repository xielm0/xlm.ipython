# -*- coding: utf-8 -*-
import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import mnist_inference

BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.01
LEARNING_RATE_DECAY = 0.99
REGULARAZTION_RATE = 0.0001
TRAINING_STEPS = 10000
MOVING_AVERAGE_DECAY = 0.99

TENSORBOARD_PATH ="/export/tensorboard_logs/"
MODEL_SAVE_PATH = "/export/tensorflow_models/"
MODEL_NAME = "mnist_distribute_synu.ckpt"
#DATA_PATH = "./mnist_tfrecords"
DATA_PATH = '/export/Data/MNIST_data'

#定义 args
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string( 'ps_hosts', 'tf-ps0:2222,tf-ps1:1111', '....')
tf.app.flags.DEFINE_string('job_name', 'worker', ' "ps" or "worker" ')
tf.app.flags.DEFINE_string( 'worker_hosts', 'tf-worker0:2222,tf-worker1:1111', '....')
tf.app.flags.DEFINE_integer( 'task_id', 0, 'Task ID of the worker/replica running the training.')

def create_optimizer( sync_flag, learning_rate, num_workers ):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    # 如果是同步模式, 通过 tf.train.SyncReplicasOptimizer函数，实现参数同步更新。
    if sync_flag:
        optimizer = tf.train.SyncReplicasOptimizer(
            # 优化器
            optimizer,
            # 每一轮更新，需要至少多少个计算服务器计算出梯度
            replicas_to_aggregate=num_workers,
            # 总共多少个计算服务器
            total_num_replicas=num_workers
        )
    return optimizer


def build_model(x, y_, n_workers):
    # L2
    regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)
    #
    y = mnist_inference.inference(x, regularizer)
    #
    global_step = tf.Variable(0, trainable=False)

    #loss
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    # learning_rate
    learning_rate = tf.train.exponential_decay(
         LEARNING_RATE_BASE,  global_step, 60000 / BATCH_SIZE, LEARNING_RATE_DECAY)

    opt = create_optimizer(True, learning_rate, n_workers )
    train_op = opt.minimize(loss, global_step=global_step)
    return global_step, loss, train_op, opt


def main(argv=None):
    # 创建集群
    ps_hosts = FLAGS.ps_hosts.split(',')
    worker_hosts = FLAGS.worker_hosts.split(',')
    n_workers = len(worker_hosts)

    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})
    server = tf.train.Server(
        cluster, job_name = FLAGS.job_name, task_index=FLAGS.task_id)

    if FLAGS.job_name == 'ps':
        server.join()
        # ps服务器运行到此为止

    # 以下worker服务器做的事情
    mnist = input_data.read_data_sets(DATA_PATH, one_hot=True)
    is_chief = (FLAGS.task_id == 0)

    # 通过tf.train.replica_device_setter自动分配device.
    device_setter = tf.train.replica_device_setter(
        worker_device="/job:worker/task:%d" % FLAGS.task_id,
        ps_device="/job:ps/cpu:0",
        cluster=cluster)
    with tf.device(device_setter):
        x = tf.placeholder(tf.float32, [None, mnist_inference.INPUT_NODE ],
            name='x-input')
        y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE],
            name='y-input')
        #
        global_step, loss, train_op, opt = build_model(x, y_, n_workers)

        saver = tf.train.Saver()
        summary_op = tf.summary.merge_all()
        init_op = tf.global_variables_initializer()

        # 在同步模式下，主计算器需要协同不同计算器计算得到的参数梯度并最终更新参数
        # 这需要在主计算服务器上完成一些额外的初始化工作
        # 异步更新参数，则不需要这一步
        if is_chief:
            # 定义协调不同计算服务器的队列，并初始化操作。
            chief_queue_runner = opt.get_chief_queue_runner()
            init_tokens_op = opt.get_init_tokens_op(0)

        # 创建会话
        sv = tf.train.Supervisor(is_chief=is_chief,
                                 logdir=TENSORBOARD_PATH,
                                 init_op=init_op,
                                 summary_op=summary_op,
                                 saver=saver,
                                 global_step=global_step,
                                 save_model_secs=60,
                                 save_summaries_secs=60)

        sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        # sess_config.gpu_options.allow_growth = True
        sess = sv.prepare_or_wait_for_session(server.target, config=sess_config)

        # 同步模式，在开始训练前，主计算服务期需要启动协调同步更新的队列，并执行初始化操作
        if is_chief:
            sv.start_queue_runners(sess, [chief_queue_runner])
            sess.run(init_tokens_op)

        step = 0
        start_time = time.time()
        while not sv.should_stop():
            # get_input
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            reshaped_xs = np.reshape(xs, (BATCH_SIZE,
                                          mnist_inference.INPUT_NODE))
            # train
            _, loss_value, global_step_value = sess.run(
                [train_op, loss, global_step], feed_dict={x: reshaped_xs, y_: ys})
            if global_step_value >= TRAINING_STEPS:
                break

            if step > 0 and step % 100 == 0:
                duration = time.time() - start_time
                sec_per_batch = duration / (global_step_value * n_workers)
                format_str = ("After %d training steps (%d global steps), "
                              "loss on training batch is %g. "
                              "(%.3f sec/batch)")
                print(format_str % (step, global_step_value,
                                    loss_value, sec_per_batch))

            step += 1

        sess.close()
        sv.stop()

if __name__ == "__main__":
    tf.app.run()

