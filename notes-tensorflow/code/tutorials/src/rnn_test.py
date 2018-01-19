import numpy as np
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import tensorflow.contrib.slim as slim


STEPS=10

def generate_data(seq):
    x=[]
    y=[]
    for i in range(len(seq)-STEPS-1):
        x.append([seq[i:i+STEPS]])
        x.append([seq[i+STEPS]])
    return np.array(x),np.array(y)

def main():
    net=slim.conv2d(input,num_outputs=32,kernel_size=[3,3])
    pass

class test(object):
    def __init__(self,
                 batch_size = 128,
                 learning_rate = 0.001,
                 error = .01,
                 display_step = 5,
                 layer_units_num = 200):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.error = error
        self.display_step = display_step
        self.layer_units_num = layer_units_num

    def dense_to_one_hot(self,labels_dense):
        """标签 转换one hot 编码
        输入labels_dense 必须为非负数
        2016-11-21
        """
        num_classes = len(np.unique(labels_dense)) # np.unique 去掉重复函数
        raws_labels = labels_dense.shape[0]
        index_offset = np.arange(raws_labels) * num_classes
        labels_one_hot = np.zeros((raws_labels, num_classes))
        labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
        return labels_one_hot

        # 获得权重和偏置
    @staticmethod
    def _weight_and_bias(in_size, out_size):
        weight = tf.truncated_normal([in_size, out_size], stddev=0.01)
        bias = tf.constant(0.1, shape=[out_size])
        return tf.Variable(weight), tf.Variable(bias)

    @lazy_property
    def length(self):
        # 返回输入稀疏张量转变为稠密张量的实际时间序列长度
        # 处理输入的稀疏矩阵，返回一个精简过的稠密矩阵描述
        # 输入稀疏矩阵 batch_size, in_lenth, in_width 输出为batch_size 对应变长时间序列 in_lenth长度
        # 输入格式为 batch_size, in_length, in_width 对in_width 取绝对值之后的最大值，如果为空置，前面默认为0，zh,则最大值返回0,
        # tf.sign 返回格式 [batch_size, in_length],其中填充数值为0,1 符号序列
        dense_sign = tf.sign(tf.reduce_max(tf.abs(self.X),axis=2))
        length = tf.reduce_sum(input_tensor=dense_sign, axis=1)
        # 返回dense_sign 的 in_length长度
        length = tf.cast(length, tf.int32)
        return length

    @staticmethod
    def _final_relevant(output, length):
        # length 输入时间序列的实际长度
        # in_length 表示输入时间序列长度
        # max_length 表示最大时间序列长度,也就是稀疏矩阵 最大时间序列长度
        batch_size = tf.shape(output)[0]
        max_length = int(output.get_shape()[1])
        output_size = int(output.get_shape()[2])
        index = tf.range(start=0, limit=batch_size)*max_length + (length-1) # 这里使用max_length 开创间隔，使用length-1表示实际位置，最后一个输出的位置
        flat = tf.reshape(output, [-1,output_size]) # 将输出展平，batch_size*length in_width
        relevant = tf.gather(flat, index) # 根据实际长度选出最后一个输出output状态使用
        return relevant

    def Preprocessing(self, trainX, trainY):
        self.in_length= in_length= trainX.shape[1]
        self.in_width= in_width= trainX.shape[2]
        self.out_classes= out_classes= trainY.shape[1]

        self.X = tf.placeholder(dtype=tf.float32, shape=[None, in_length, in_width], name='trainX') # 批次，时间序列，多因子
        self.Y = tf.placeholder(dtype= tf.float32, shape=[None, out_classes], name='trainY')
        self.keep_prob = tf.placeholder(dtype= tf.float32)

    def str2float(self,s):
        def char2num(s):
            return {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9}[s]
        n = s.index('.')
        return reduce(lambda x,y:x*10+y,map(char2num,s[:n]+s[n+1:]))/(10**n)

    def Interface(self):
        # 4层GRU结构描述
        monolayer = tf.nn.rnn_cell.GRUCell(num_units= self.layer_units_num)
        monolayer = tf.nn.rnn_cell.DropoutWrapper(cell=monolayer, output_keep_prob=self.keep_prob)
        monolayer_final = tf.nn.rnn_cell.GRUCell(num_units= self.layer_units_num)
        layers = tf.nn.rnn_cell.MultiRNNCell([monolayer]*3+[monolayer_final])
        # 激活 注意 in_length 表示输入序列步长， length 表示实际步长
        output,_ = tf.nn.dynamic_rnn(cell= layers, inputs= self.X, dtype= tf.float32, sequence_length= self.length)
        output = self._final_relevant(output, self.length)

        weights, biases = self._weight_and_bias(self.layer_units_num, self.out_classes)
        Prediction = tf.nn.bias_add(tf.matmul(output, weights),biases)
        return Prediction

    def Graph(self, trainX, trainY):
        try:
            tf.InteractiveSession.close()
        except:
            pass
        self.sess = tf.InteractiveSession()
        tf.get_default_session()
        self.Preprocessing(trainX, trainY)
        tmp = self.Interface()

        self.pred = tf.nn.softmax(tmp)
        self.cost = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(tmp, self.Y))

        optimizer = tf.train.AdamOptimizer(learning_rate= self.learning_rate) # 0 设置训练器
        grads_and_vars = optimizer.compute_gradients(self.cost)
        for i, (grid, var) in enumerate(grads_and_vars):
            if grid != None:
                grid = tf.clip_by_value(grid, -1., 1.)
                grads_and_vars[i] = (grid, var)
        optimizer = optimizer.apply_gradients(grads_and_vars)
        self.optimizer = optimizer

        self.correct_pred = tf.equal(tf.argmax(tmp,1), tf.argmax(self.Y,1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))
        self.init = tf.global_variables_initializer()

    def fit(self, trainX, trainY, dropout= 0.618):
        # 对标签 one_hot编码
        trainY = self.dense_to_one_hot(trainY)

        self.Graph(trainX, trainY)
        self.sess.run(self.init)
        batch_size = self.batch_size
        loss =1000.
        ep = 0
        while (loss > self.error):
            for i in range(int(len(trainX)/batch_size)+1):
                if i < int(len(trainX)/batch_size)+1:
                    batch_x = trainX[i*batch_size : (i+1)*batch_size]
                    batch_y = trainY[i*batch_size : (i+1)*batch_size]
                elif i== int(len(trainX)/batch_size)+1:
                    batch_x = trainX[-batch_size:]
                    batch_y = trainY[-batch_size:]
                self.sess.run(self.optimizer,feed_dict={self.X:batch_x, self.Y:batch_y, self.keep_prob:(1.-dropout)})
            loss = self.sess.run(self.cost, feed_dict={self.X:trainX, self.Y:trainY, self.keep_prob:1.})
            if ep%self.display_step==0:
                acc = self.sess.run(self.accuracy, feed_dict={self.X:trainX, self.Y:trainY, self.keep_prob:1.})
                print (str(ep)+"th "+'Epoch Loss = {:.5f}'.format(loss)+" Training Accuracy={:.5f}".format(acc))
            ep += 1
        print("Optimization Finished!")

    def pred_prob(self, testX):
        batch_size = self.batch_size
        trainX = testX
        predict_output = np.zeros([1,self.out_classes])
        for i in range(int(len(trainX)/batch_size)+1):
            if i < int(len(trainX)/batch_size)+1:
                batch_x = trainX[i*batch_size : (i+1)*batch_size]
                batch_y = trainY[i*batch_size : (i+1)*batch_size]
            elif i== int(len(trainX)/batch_size)+1:
                batch_x = trainX[-batch_size:]
                batch_y = trainY[-batch_size:]
            tp = self.sess.run(self.pred, feed_dict={self.X:batch_x, self.keep_prob:1.})
            predict_output = np.row_stack([predict_output, tp])
        predict_output = np.delete(predict_output, obj=0, axis=0)
        return predict_output

    def pred_signal(self, testX):
        pred_prob = self.pred_prob(testX)
        return np.argmax(pred_prob, axis=1)