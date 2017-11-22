# -*- coding: utf-8 -*-
import time

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

import mnist_inference_cnn

BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.01
LEARNING_RATE_DECAY = 0.99
REGULARAZTION_RATE = 0.0001
TRAINING_STEPS = 10000
MOVING_AVERAGE_DECAY = 0.99
MODEL_SAVE_PATH = "./distribute_syn_multi_gpu_oneTaskOneGPU_models"
DATA_PATH = '/export/Data/MNIST_data'

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('job_name', 'worker', ' "ps" or "worker" ')
tf.app.flags.DEFINE_string(
    'ps_hosts', ' tf-ps0:2222,tf-ps1:1111',
    'Comma-separated list of hostname:port for the parameter server jobs. '
    'e.g. "tf-ps0:2222,tf-ps1:1111"')
tf.app.flags.DEFINE_string(
    'worker_hosts', ' tf-worker0:2222,tf-worker1:1111',
    'Comma-separated list of hostname:port for the worker jobs. '
    'e.g. "tf-worker0:2222,tf-worker1:1111"')

tf.app.flags.DEFINE_integer(
    'task_id', 0, 'Task ID of the worker/replica running the training.')

tf.app.flags.DEFINE_integer(
    'gpu_id', 0, 'Task ID of the worker/replica running the training.')




def build_model(x, y_, n_worker):
    regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)
    y = mnist_inference_cnn.inference(x, True, regularizer)
    global_step = tf.Variable(0, trainable=False)

    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    # learning_rate = tf.train.exponential_decay(
    #     LEARNING_RATE_BASE,
    #     global_step,
    #     60000 / BATCH_SIZE,
    #     LEARNING_RATE_DECAY)
    learning_rate = LEARNING_RATE_BASE

    opt = tf.train.SyncReplicasOptimizer(
        tf.train.GradientDescentOptimizer(learning_rate),
        replicas_to_aggregate=n_worker,
        total_num_replicas=n_worker,
        # replica_id=FLAGS.task_id)
    )
    apply_gradient_op = opt.minimize(loss, global_step=global_step)
    train_op = tf.group(apply_gradient_op, variables_averages_op)
    return global_step, loss, train_op, opt


def main(argv=None):
    ps_hosts = FLAGS.ps_hosts.split(',')
    worker_hosts = FLAGS.worker_hosts.split(',')
    n_workers = len(worker_hosts)

    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})
    server = tf.train.Server(
        cluster, job_name = FLAGS.job_name, task_index=FLAGS.task_id)

    if FLAGS.job_name == 'ps':
        server.join()


    is_chief = (FLAGS.task_id == 0)
    mnist = input_data.read_data_sets(DATA_PATH, one_hot=True)

    with tf.device(
            tf.train.replica_device_setter(
                worker_device="/job:worker/task:%d/gpu:%d" % (FLAGS.task_id, FLAGS.gpu_id),
                ps_device="/job:ps/cpu:0",
                cluster=cluster)):
        x = tf.placeholder(tf.float32, [
            None,
            mnist_inference_cnn.IMAGE_SIZE,
            mnist_inference_cnn.IMAGE_SIZE,
            mnist_inference_cnn.NUM_CHANNELS],
            name='x-input')
        y_ = tf.placeholder(
            tf.float32, [None, mnist_inference_cnn.OUTPUT_NODE],
            name='y-input')
        global_step, loss, train_op, opt = build_model(
            x, y_, n_workers)

        saver = tf.train.Saver()
        summary_op = tf.summary.merge_all()
        init_op = tf.initialize_all_variables()

        # 在同步模式下,主计算服务器需要协调不同计算服务器计算得到所有的参数梯度并最终更新参数。这需要主计算服务器完成一些额外的初始化操作
        if is_chief:
            chief_queue_runner = opt.get_chief_queue_runner()
            init_tokens_op = opt.get_init_tokens_op(0)

        sv = tf.train.Supervisor(is_chief=is_chief,
                                 logdir=MODEL_SAVE_PATH,
                                 init_op=init_op,
                                 summary_op=summary_op,
                                 saver=saver,
                                 global_step=global_step,
                                 save_model_secs=60,
                                 save_summaries_secs=60)
        sess_config = tf.ConfigProto(allow_soft_placement=True,
                                     log_device_placement=False)
        sess = sv.prepare_or_wait_for_session(
            server.target, config=sess_config)

        if is_chief:
            sv.start_queue_runners(sess, [chief_queue_runner])
            sess.run(init_tokens_op)

        step = 0
        start_time = time.time()
        while not sv.should_stop():
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            reshaped_xs = np.reshape(xs, (BATCH_SIZE,
                                          mnist_inference_cnn.IMAGE_SIZE,
                                          mnist_inference_cnn.IMAGE_SIZE,
                                          mnist_inference_cnn.NUM_CHANNELS))
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
        sv.stop()


if __name__ == "__main__":
    tf.app.run()

