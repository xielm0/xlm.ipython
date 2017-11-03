# -*- coding:utf-8 -*-
import time

import tensorflow as tf

import mnist_inference2


# 定义训练神经网络需要用到的配置
BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.0001
LEARNING_RATE_DECAY = 0.99
REGULARAZTION_RATE = 0.0001
TRAINING_STEPS = 1000
MOVING_AVERAGE_DECAY = 0.99
N_GPU = 4

# 定义日志和模型输出的路径
MODEL_SAVE_PATH = "./distribute_syn_multi_gpu_models/"
DATA_PATH = "./mnist_tfrecords"


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




def get_input():
    filename_queue = tf.train.string_input_producer([DATA_PATH])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
        serialized_example,
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'pixels': tf.FixedLenFeature([], tf.int64),
            'label': tf.FixedLenFeature([], tf.int64),
        })
    decoded_image = tf.decode_raw(features['image_raw'], tf.uint8)
    reshaped_image = tf.reshape(decoded_image, [28, 28, 1])
    retyped_image = tf.cast(reshaped_image, tf.float32)
    label = tf.cast(features['label'], tf.int32)

    min_after_dequeue = 10000
    capacity = min_after_dequeue + 3 * BATCH_SIZE
    return tf.train.shuffle_batch(
        [retyped_image, label],
        batch_size=BATCH_SIZE,
        capacity=capacity,
        min_after_dequeue=min_after_dequeue)


def get_loss(x, y_, regularizer, scope):
    y = mnist_inference2.inference(x, train=True, regularizer=regularizer)
    cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=y_))
    regularization_loss = tf.add_n(tf.get_collection('losses', scope))
    loss = cross_entropy + regularization_loss
    return loss


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
    ps_hosts = FLAGS.ps_hosts.split(',')
    worker_hosts = FLAGS.worker_hosts.split(',')
    n_workers = len(worker_hosts)

    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})
    server = tf.train.Server(
        cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_id)

    if FLAGS.job_name == 'ps':
        server.join()
        # ps服务器运行到此为止


    # 以下worker服务器做的事情
    is_chief = (FLAGS.task_id == 0)
    worker_device = "/job:worker/task:%d" % FLAGS.task_id
    with tf.Graph().as_default(), tf.device(
            tf.train.replica_device_setter(
                worker_device=worker_device,
                ps_device="/job:ps/cpu:0",
                cluster=cluster)):
        x, y_ = get_input()
        regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)
        global_step = tf.get_variable(
            'global_step', [], initializer=tf.constant_initializer(0),
            trainable=False)
        # learning_rate = tf.train.exponential_decay(
        #     LEARNING_RATE_BASE, global_step, 60000 / BATCH_SIZE,
        #     LEARNING_RATE_DECAY
        # )
        learning_rate = LEARNING_RATE_BASE


        grad_opt = tf.train.GradientDescentOptimizer(learning_rate)

        tower_grads = []

        # 记录每个GPU的损失函数值
        loss_gpu_dir = {}
        #

        # 将神经网络的优化过程跑在不同的GPU上
        for i in xrange(N_GPU):
            worker_device = "/job:worker/task:%d/gpu:%d" % (FLAGS.task_id, i)
            with tf.device(
                tf.train.replica_device_setter(
                    worker_device=worker_device,
                    ps_device="/job:ps/cpu:0",
                    cluster=cluster)):
            # with tf.device("/job:worker/task:%d/gpu:%d" % (FLAGS.task_id, i)):
                with tf.name_scope('GPU_%d' % i) as scope:

                    cur_loss = get_loss(x, y_, regularizer, scope)
                    #
                    loss_gpu_dir['GPU_%d' % i] = cur_loss
                    #
                    tf.get_variable_scope().reuse_variables()
                    grads = grad_opt.compute_gradients(cur_loss)
                    tower_grads.append(grads)
        #
        for los in loss_gpu_dir:
            tf.summary.scalar('GPU Loss/' + los, loss_gpu_dir[los])
        #

        # 计算变量的平均梯度，并输出到TensorBoard日志
        grads = average_gradients(tower_grads)
        for grad, var in grads:
            if grad is not None:
                tf.summary.histogram(
                    'gradients_on_average/%s' % var.op.name, grad)
        # 使用平均梯度更新参数
        tf.get_variable_scope()._reuse = False

        opt = tf.train.SyncReplicasOptimizer(
            grad_opt,
            replicas_to_aggregate=n_workers,
            total_num_replicas=n_workers,
            # replica_id=FLAGS.task_id)
            use_locking=True)

        #### debug
        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
        ####
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)

        # 计算变量的滑动平均值
        variable_averages = tf.train.ExponentialMovingAverage(
            MOVING_AVERAGE_DECAY, global_step)
        variables_averages_op = variable_averages.apply(
            tf.trainable_variables())
        tf.get_variable_scope()._reuse = True

        train_op = tf.group(apply_gradient_op, variables_averages_op)

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
        # with sv.prepare_or_wait_for_session(
        #     server.target, config=sess_config) as sess:
        if is_chief:
            sv.start_queue_runners(sess, [chief_queue_runner])
            sess.run(init_tokens_op)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        step = 0
        start_time = time.time()

        while not sv.should_stop():
            _, global_step_value, loss_value1, loss_value2, loss_value3, loss_value4 = sess.run(
                [train_op, global_step] + loss_gpu_dir.values())

            if global_step_value >= TRAINING_STEPS:
                break
            if step > 0 and step % 100 == 0:
                duration = time.time() - start_time
                sec_per_batch = duration / (global_step_value * n_workers)
                format_str = ("After %d training steps (%d global steps), "
                                "loss on training batch is %g, %g, %g, %g. "
                                "(%.3f sec/batch)")
                print(format_str % (step, global_step_value,
                                    loss_value1, loss_value2,
                                    loss_value3, loss_value4,
                                    sec_per_batch))
            step += 1
        coord.request_stop()
        coord.join(threads)
        sess.close()
        sv.stop()


if __name__ == "__main__":
    tf.app.run()




