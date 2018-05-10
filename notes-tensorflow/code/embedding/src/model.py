# -*- coding:utf-8 -*-
import tensorflow as tf
import math
# import six.moves.reduce as reduce


class Options(object):
    def __init__(self):
        # model params
        self.hash_size=[100,1000,1000,10**5,10**5,10**5]
        self.embedding_size=[8,10,14,32,32,32]  # sum=128
        self.vocabulary_size=int(1e5)
        self.num_sampled=32

class sku2vec(object):
    def __init__(self,
                 param_dict,
                 **kwargs):
        self.param_dict = param_dict


    def get_weight_variable(self, shape, initializer, regularizer=None):
        weights = tf.get_variable("weights", shape, initializer=initializer, dtype=tf.float32)
        if regularizer != None:
            tf.add_to_collection('losses', regularizer(weights))
        return weights

    def inference(self,x,train_flag=None):
        x_size = self.param_dict["hash_size"]
        embedding_size = self.param_dict["embedding_size"]
        feature_nums = len(x_size)

        embed_list = []
        for i in range(feature_nums):
            with tf.variable_scope('embedding_%d' % i):
                init_with=0.5/embedding_size[i]
                EmbedWeights = self.get_weight_variable(shape=[x_size[i], embedding_size[i]],
                                                   initializer=tf.random_uniform_initializer(-init_with, init_with))
                embed_i = tf.nn.embedding_lookup(EmbedWeights, tf.cast(x[:, i], tf.int32))
                embed_list.append(embed_i)

        embed = reduce(lambda a, b: tf.concat([a, b], 1), embed_list)
        return embed


    def get_loss(self,inputs, labels):
        num_sampled=self.param_dict["num_sampled"]
        vocabulary_size= self.param_dict["vocabulary_size"]

        embed = self.inference(inputs,"train")

        # Construct the variables for the NCE loss
        # 单GPU 可以 f.Variable
        # nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embed.shape[1].value],
        #                         stddev=1.0 / math.sqrt(embed.shape[1].value)))
        # nce_biases = tf.Variable(tf.zeros([vocabulary_size]))
        # 当使用多GPU时,要使用tf.get_variable
        nce_weights = tf.get_variable("nce_weights",shape=[vocabulary_size, embed.shape[1].value],
                                      initializer=tf.truncated_normal_initializer(stddev=1.0 / math.sqrt(embed.shape[1].value)))
        nce_biases = tf.get_variable("nce_biases",shape =[vocabulary_size],
                                     initializer=tf.constant_initializer(0.0))
        loss = tf.reduce_mean(
            tf.nn.nce_loss(weights=nce_weights,
                           biases=nce_biases,
                           labels=labels,
                           inputs=embed,
                           num_sampled=num_sampled,
                           num_classes=vocabulary_size))

        # regularization_loss = tf.add_n(tf.get_collection("losses", scope))
        with tf.name_scope("loss"):
            tf.summary.scalar("loss", loss)

        return loss



