# -*- coding:utf-8 -*-
from datetime import datetime
import os
import time
import tensorflow as tf
import mnist_inference

# 定义训练神经网络需要用到的配置
BATCH_SIZE = 128
LEARNING_RATE_BASE = 0.0001
LEARNING_RATE_DECAY = 0.99
REGULARAZTION_RATE = 0.0001
TRAINING_STEPS = 1000
MOVING_AVERAGE_DECAY = 0.99
N_GPU = 4

# 定义路径
TENSORBOARD_PATH ="/export/tensorboard_logs/"
MODEL_SAVE_PATH = "/export/tensorflow_models/"
MODEL_NAME = "mnist_multi_gpu.ckpt"
DATA_PATH = "./mnist_tfrecords"

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
    reshaped_image = tf.reshape(decoded_image, [784])
    retyped_image = tf.cast(reshaped_image, tf.float32)
    label = tf.cast(features['label'], tf.int32)

    min_after_dequeue = 10000
    capacity = min_after_dequeue + 3 * BATCH_SIZE
    return tf.train.shuffle_batch(
        [retyped_image, label],  # [784,10]
        batch_size=BATCH_SIZE,
        capacity=capacity,
        min_after_dequeue=min_after_dequeue)

def get_loss(x, y_, regularizer, scope):
    y = mnist_inference.inference(x, regularizer=regularizer)
    cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=y_))
    regularization_loss = tf.add_n(tf.get_collection('losses', scope))
    loss = cross_entropy + regularization_loss
    return loss

# 计算每一个变量的平均梯度
def average_gradients(tower_grads):
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


def main(argv=None):
    # 将简单运算放在CPU上,只有神经网络的训练放在GPU上。
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        # 获取一个batch的数据,x [100, 784]
        x, y_ = get_input()
        regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)

        global_step = tf.get_variable(
            'global_step', [], initializer=tf.constant_initializer(0),
            trainable=False)

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
                    cur_loss = get_loss(x, y_, regularizer, scope)
                    loss_gpu_dir['GPU_%d' % i] = cur_loss
                    # 为了不同gpu更新同一组组参数，需要将
                    tf.get_variable_scope().reuse_variables()
                    # 当前GPU计算当前梯度。
                    grads = opt.compute_gradients(cur_loss)
                    tower_grads.append(grads)

        # 写tensorboard日志
        tf.summary.scalar('loss',cur_loss)

        # 计算平均梯度，并输出到TensorBoard日志
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
        variable_averages = tf.train.ExponentialMovingAverage( MOVING_AVERAGE_DECAY, global_step)
        variables_averages_op = variable_averages.apply(tf.trainable_variables())


        train_op = tf.group(apply_gradient_op, variables_averages_op)

        saver = tf.train.Saver(tf.all_variables())
        summary_op = tf.summary.merge_all()

        init = tf.initialize_all_variables()

        with tf.Session(config=tf.ConfigProto( allow_soft_placement=True, log_device_placement=True)
                        ) as sess:
            init.run()
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            summary_writer = tf.summary.FileWriter( TENSORBOARD_PATH, sess.graph)

            for step in range(TRAINING_STEPS):
                start_time = time.time()
                _, loss_value = sess.run([train_op, cur_loss])
                duration = time.time() - start_time

                if step != 0 and step % 10 == 0:
                    num_examples_per_step = BATCH_SIZE * N_GPU
                    examples_per_sec = num_examples_per_step / duration
                    sec_per_batch = duration / N_GPU

                    format_str = ('step %d, loss = %.2f (%.1f examples sec; %.3f sec/batch)')
                    print(format_str % (step, loss_value,  examples_per_sec, sec_per_batch))

                    # 写tensorboard日志
                    summary = sess.run(summary_op)
                    summary_writer.add_summary(summary, step)

                    if step % 1000 == 0 or (step + 1) == TRAINING_STEPS:
                        checkpoint_path = os.path.join(MODEL_SAVE_PATH, MODEL_NAME)
                        saver.save(sess, checkpoint_path, global_step=step)

            coord.request_stop()
            coord.join(threads)

if __name__ == '__main__':
    tf.app.run()

