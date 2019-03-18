# -*- coding: UTF-8 -*-

import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.model_selection  import train_test_split
import logging
import os
from sklearn import metrics

flags = tf.app.flags
flags.DEFINE_integer("batch_size", 10 , "batch_size" )

FLAGS = flags.FLAGS


# model
MODEL_SAVE_PATH = "../models/"
MODEL_NAME = "model.ckpt"

class Params(object):
    def __init__(self):
        # model params
        self.batch_size=FLAGS.batch_size
        self.embedding_size=5
        self.learning_rate=0.0001
        self.lambda1 =0.01
        self.lambda2 =0.0001



class Fm(object):
    def __init__(self, params ):
        self.batch_size=params.batch_size
        self.k=params.embedding_size
        self.learning_rate=params.learning_rate
        self.lambda1 = params.lambda1
        self.lambda2 = params.lambda2


    def get_weight_variable(self, name, shape, initializer, regularizer=None):
        weights = tf.get_variable(name, shape, initializer=initializer, dtype=tf.float32)
        if regularizer != None:
            tf.add_to_collection('losses', regularizer(weights))
        return weights

    def cross(self,x,share_EmbedWeights):
        """
        先x的每个特征进行embedding得到vi，然后 xi*vi ,然后再求内积
        目标：[batch, n] * [n,k] =[batch,n,k]
        转化为： [n*[batch, 1]] * [n*[batch,k]] =[batch,n,k], x的每一行要转化成一个对角矩阵
        """
        n=x.shape[1].value #feature nums
        k=self.k
        #
        embeds = []
        for i in range(n):
            xi = x[:,i] # shape=[batch]
            vi = tf.nn.embedding_lookup(share_EmbedWeights, i) #shape=[k]
            # 直接 xi*vi 报错
            # xi*v_i=[batch,1] * [1,k]=[batch,k]
            embed_i = tf.expand_dims(xi,1) * tf.expand_dims(vi ,0)
            embeds.append(embed_i)

        embed = tf.reshape(tf.concat(embeds, 1),[-1,n,k])  # shape=[-1,n,k]
        #
        # sum_square 表示先sum，square. sum是axis=1，而不是axis=2
        sum_square = tf.square(tf.reduce_sum(embed, axis=1))
        square_sum = tf.reduce_sum(tf.square(embed), axis=1)
        y_v = 0.5 * tf.reduce_sum(sum_square - square_sum, axis=1)

        return y_v

    def cross2(self,x1,x2,x1_pos,x2_pos,EmbedWeights):
        # 这里xi是数字，则按位置取embedding,如果是onehot更简单
        v1 = tf.nn.embedding_lookup(EmbedWeights, x1_pos) #shape=[k]
        v2 = tf.nn.embedding_lookup(EmbedWeights, x2_pos)
        # x1 * v1 =[batch,1] * [1,k]=[batch,k]
        embed_1 = tf.expand_dims(x1, 1) * tf.expand_dims(v1, 0)
        embed_2 = tf.expand_dims(x2, 1) * tf.expand_dims(v2, 0)
        # 内积 [batch,1,k] * [batch,k,1] = [batch,1,1]
        y=tf.matmul(tf.expand_dims(embed_1,1),tf.expand_dims(embed_2,2))
        y=tf.squeeze(y)
        return y



    def cross_layer(self,x0,x,w,b):
        """
        deep cross layer
        x0*x=[n,1]*[1,n]=[n,n]
        x0*x *w = [n,n] * [n,1]=[n,1]
        :param x0: shape=[batch,n] eg: x0 = tf.ones(shape=[10,3])
        :param x: shape=[batch,n]  eg: x = tf.ones(shape=[10, 3])
        :param w: shape=[n]        eg: w= tf.constant([1.,2.,3.])
        :return : shape=[batch,n]
        """
        # batch = x0.shape[0].value
        batch=self.batch_size
        n = x0.shape[1].value  # feature nums
        # [batch,n,1] * [batch,1,n]= [batch,n,n]
        m = tf.matmul(tf.expand_dims(x0, 2), tf.expand_dims(x,1)) #[batch,n,n]
        # [batch,n,n] * [batch,n,1]= [batch,n,1]
        new_w=tf.reshape(tf.tile(tf.expand_dims(w,1),[batch,1]),[-1,n,1])  #[batch,n,1]
        y= tf.squeeze(tf.matmul(m,new_w))+x+b
        return  y

    def cross_dcn(self,x0,x,w,b):
        """
        使用矩阵运算规则，进行计算优化: x0*x'*w= x0* (x'*w)
        即[n,1]*[1,n]*[n,1]=[n,1]*([1,n]*[n,1])=[n,1]*[1,1]=[n,1]
        其中，[1,n]*[n,1]等同向量内积
        :param x0: shape=[batch,n] eg: x0 = tf.ones(shape=[10,3])
        :param x: shape=[batch,n]  eg: x = tf.ones(shape=[10, 3])
        :param w: shape=[n]        eg: w= tf.constant([1.,2.,3.])
        :return : shape=[batch,n]
        """
        # [batch,n]*[n,1]= [batch,1]
        new_w=tf.reshape(w,[-1,1])
        inner= tf.matmul(x,new_w)
        # [batch,n] * [batch,1]= [batch,n]  # 使用 乘法，而不是矩阵乘法
        y= x0*inner + x + b
        return  y


    def inference(self,x):
        n= x.shape[1].value  #feature nums
        k= self.k
        # b
        w0 = tf.get_variable("w0", [1],
                             initializer=tf.constant_initializer(0.0), dtype=tf.float32)
        # 一次项
        w1=tf.get_variable("w1", shape=[n, 2],
                           initializer=tf.random_uniform_initializer(-0.5/n, 0.5/n))
        # 二次项
        w2 = tf.get_variable("w2", shape=[n, k],
                             initializer=tf.truncated_normal_initializer(stddev= 1.0/tf.sqrt(float(n))))

        y1 = w0+ tf.reduce_sum(tf.matmul(x , w1), axis=1)

        # 二次项
        y_v = self.cross(x,w2)
        #y_v =self.cross2(x[:,1],x[:,2],1,2,w2)
        #
        y_fm = y1 + y_v
        return y_fm

    def inference_dcn(self,x):
        n = x.shape[1].value  # feature nums
        with tf.variable_scope('cross_layer_1'):
            b=tf.get_variable("b", [n],initializer=tf.constant_initializer(0.0), dtype=tf.float32)
            w=tf.get_variable("w", shape=[n],
                               initializer=tf.random_uniform_initializer(-0.5/n, 0.5/n))
            x1 =self.cross_layer(x,x,w,b)

        with tf.variable_scope('cross_layer_2'):
            b=tf.get_variable("b", [n],initializer=tf.constant_initializer(0.0), dtype=tf.float32)
            w=tf.get_variable("w", shape=[n],
                               initializer=tf.random_uniform_initializer(-0.5/n, 0.5/n))
            x2 =self.cross_layer(x,x1,w,b)

        with tf.variable_scope('layer_1'):
            b = tf.get_variable("b", [3], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
            w = tf.get_variable("w", shape=[n,3],
                               initializer=tf.random_uniform_initializer(-0.5/n, 0.5/n))
            layer2 = tf.nn.relu(tf.matmul(x, w) + b)
            tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(self.lambda1)(w))

        with tf.variable_scope('output'):
            x_stack = tf.concat([x2,layer2],axis=1)
            w = tf.get_variable("w", shape=[x_stack.shape[1].value, 1],
                                initializer=tf.random_uniform_initializer(-0.5 / n, 0.5 / n))
            b =tf.get_variable("b", [1], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
            y= tf.sigmoid(tf.matmul(x_stack,w) + b)
            return y

    def get_loss(self,label,pred,scope=None):
        loss = tf.reduce_mean(0.5 * tf.square(label - pred))
        # loss =-tf.reduce_mean( label * tf.log(tf.clip_by_value(pred, 1e-10, 1.0)) + (1 - label) * tf.log(tf.clip_by_value(1 - pred, 1e-10, 1.0)))
        # reg_loss = tf.add_n(tf.get_collection("losses",scope))
        # loss= loss + reg_loss
        return loss


    def get_optimizer(self,optimizer, learning_rate):
        logging.info("Use the optimizer: {}".format(optimizer))
        if optimizer == "sgd":
            return tf.train.GradientDescentOptimizer(learning_rate)
        elif optimizer == "adadelta":
            return tf.train.AdadeltaOptimizer(learning_rate)
        elif optimizer == "adagrad":
            return tf.train.AdagradOptimizer(learning_rate)
        elif optimizer == "adam":
            return tf.train.AdamOptimizer(learning_rate)
        elif optimizer == "ftrl":
            return tf.train.FtrlOptimizer(learning_rate)
        elif optimizer == "rmsprop":
            return tf.train.RMSPropOptimizer(learning_rate)
        else:
            logging.error("Unknow optimizer, exit now")
            exit(1)

    def train_with_one_gpu(self,x_,y_):
        # y=self.inference(x_)
        y=self.inference_dcn(x_)
        loss=self.get_loss(y_,y)
        opt=self.get_optimizer("sgd",self.learning_rate)
        grads_and_vars = opt.compute_gradients(loss)
        train_op = opt.apply_gradients(grads_and_vars )
        return train_op,loss


def get_input():
    # data load
    iris = load_iris()
    x = iris["data"]
    y = iris["target"]

    x, y = x[y != 2], y[y != 2]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=123)

    return x_train, x_test, y_train, y_test


def train():

    x_train, x_test, y_train, y_test = get_input()
    feature_nums= 4
    # 计算batch_num
    n = len(x_train)
    batch_size=FLAGS.batch_size
    batch_num = int(n / batch_size) if n % batch_size == 0 else int(n / batch_size) + 1
    print("n= %s , epoch_num= %s " % (n, batch_num))


    x_ = tf.placeholder(tf.float32, [None, feature_nums] ,name='x-input')
    y_ = tf.placeholder(tf.float32, [None] , name='y-input')

    params=Params()
    fm =Fm(params)
    train_op,loss = fm.train_with_one_gpu(x_,y_)
    #
    saver = tf.train.Saver()
    init = (tf.global_variables_initializer(), tf.local_variables_initializer())

    CONFIG = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    CONFIG.gpu_options.allow_growth = True
    with tf.Session(config=CONFIG) as sess:
        sess.run(init)
        for epoch in range(30):
            for i in range(batch_num):
                a = i * batch_size
                b = (i + 1) * batch_size
                b = b if b <= n else n
                x_batch,y_batch =x_train[a:b,] ,y_train[a:b,]
                _, loss_vlaue = sess.run([train_op, loss],feed_dict={x_: x_batch, y_: y_batch})

            format_str = ('epoch %d,batch %d, loss = %.10f  ')
            print(format_str % (epoch, i, loss_vlaue ))

            # save graph ,include vars
            step=epoch*batch_num +i
            # if step % 10 == 0 or (step + 1) == max_steps:
            checkpoint_path = os.path.join(MODEL_SAVE_PATH,MODEL_NAME)
            saver.save(sess, checkpoint_path, global_step=step)



def eval():
    x_train, x_test, y_train, y_test = get_input()
    x_test = x_test[0:10, ]
    y_test = y_test[0:10, ]
    feature_nums = 4
    x_ = tf.placeholder(tf.float32, [None, feature_nums], name='x-input')
    y_ = tf.placeholder(tf.float32, [None], name='y-input')
    params = Params()
    fm = Fm(params)
    tf.get_variable_scope()._reuse=True
    pred = fm.inference_dcn(x_)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            # auc
            y_pred = sess.run(pred, feed_dict={x_: x_test, y_: y_test})
            auc_scroe = eval_metrics(y_test, y_pred)
            print("After %s step , AUC score = %g" % (global_step, auc_scroe))




def eval_metrics(labels,pred):
    fpr, tpr, thresholds = metrics.roc_curve(labels, pred)
    auc_scroe = metrics.auc(fpr, tpr)
    # acc_score = metrics.accuracy_score(labels,pred)
    # print(acc_score)
    return auc_scroe

if __name__ == '__main__':
    train()
    eval()