# -*- coding: utf-8 -*-

"""
==================================================
   File Name：     3.使用循环神经网络实现语言模型
   email:         songdongdong@jd.com
   Author :       songdongdong
   date：          2019/3/24
   Description :  
==================================================
"""
__author__ = 'songdongdong'
import numpy as np
import tensorflow as tf
import reader_ptb


DATA_PATH = "../../datasets/PTB_data/" #存数据路径
HIDDEN_SIZE = 200   #隐藏层规模
NUM_LAYERS = 2      #深层循环神经网络中LSTM结构的层数
VOCAB_SIZE = 10000  #单词标识符总共一万个单词

LEARNING_RATE = 1.0    #学习速率
TRAIN_BATCH_SIZE = 20  #选了数据batch的大小
TRAIN_NUM_STEP = 35   #训练数据截断的长度

#在测试时候不需要使用截断，所以可以将测试数据看成一个超长的序列
EVAL_BATCH_SIZE = 1      #测试数据batch的大小
EVAL_NUM_STEP = 1        #测试数据截断长度
NUM_EPOCH = 2            #使用训练数据的轮数
KEEP_PROB = 0.5         #节点不被dropput的概率
MAX_GRAD_NORM = 5       #用于控制梯度膨胀的参数

# 2.定义一个类来描述模型结构。
class PTBModel(object):
    def __init__(self, is_training, batch_size, num_steps):

        self.batch_size = batch_size  #batch大小
        self.num_steps = num_steps  #截断长度

        # 定义输入层。
        self.input_data = tf.placeholder(tf.int32, [batch_size, num_steps])
        self.targets = tf.placeholder(tf.int32, [batch_size, num_steps])

        # 定义使用LSTM结构及训练时使用dropout。
        # lstm_cell = tf.contrib.rnn.BasicLSTMCell(HIDDEN_SIZE,)
        # #
        # if is_training:
        #     lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=KEEP_PROB)
        # cell = tf.contrib.rnn.MultiRNNCell([lstm_cell for _ in range(0 ,NUM_LAYERS)], ) #和原版有改动
        #用上面的代码会报错，原因：这里每次都需要新创建的lstm_cell,不能重复使用
        cell = tf.contrib.rnn.MultiRNNCell([self.lstm_cell(HIDDEN_SIZE, keep_prob=KEEP_PROB,is_training=is_training) for _ in range(NUM_LAYERS)],
                                          state_is_tuple=True)


        # 初始化最初的状态。
        self.initial_state = cell.zero_state(batch_size, tf.float32)
        embedding = tf.get_variable("embedding", [VOCAB_SIZE, HIDDEN_SIZE])

        # 将原本单词ID转为单词向量。
        inputs = tf.nn.embedding_lookup(embedding, self.input_data)

        if is_training:
            inputs = tf.nn.dropout(inputs, KEEP_PROB)

        # 定义输出列表。
        outputs = []
        state = self.initial_state
        with tf.variable_scope("RNN"):
            for time_step in range(num_steps):
                if time_step > 0: tf.get_variable_scope().reuse_variables()
                cell_output, state = cell(inputs[:, time_step, :], state)
                outputs.append(cell_output)
        output = tf.reshape(tf.concat(outputs, 1), [-1, HIDDEN_SIZE])
        weight = tf.get_variable("weight", [HIDDEN_SIZE, VOCAB_SIZE])
        bias = tf.get_variable("bias", [VOCAB_SIZE])
        logits = tf.matmul(output, weight) + bias

        # 定义交叉熵损失函数和平均损失。
        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
            [logits],
            [tf.reshape(self.targets, [-1])],
            [tf.ones([batch_size * num_steps], dtype=tf.float32)])
        self.cost = tf.reduce_sum(loss) / batch_size
        self.final_state = state

        # 只在训练模型时定义反向传播操作。
        if not is_training: return
        trainable_variables = tf.trainable_variables()

        # 控制梯度大小，定义优化方法和训练步骤。
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, trainable_variables), MAX_GRAD_NORM)
        optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE)
        self.train_op = optimizer.apply_gradients(zip(grads, trainable_variables))

    def lstm_cell(self,hidden_size, keep_prob,is_training):

        cell = tf.contrib.rnn.BasicLSTMCell(hidden_size, reuse=tf.get_variable_scope().reuse) #LSTMCELL
        if is_training:
            return tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)
        return cell


#3. 使用给定的模型model在数据data上运行train_op并返回在全部数据上的perplexity值
def run_epoch(session, model, data, train_op, output_log, epoch_size):
    total_costs = 0.0
    iters = 0
    state = session.run(model.initial_state)

    # 训练一个epoch。
    for step in range(epoch_size):
        x, y = session.run(data)
        cost, state, _ = session.run([model.cost, model.final_state, train_op],
                                        {model.input_data: x, model.targets: y, model.initial_state: state})
        total_costs += cost
        iters += model.num_steps

        if output_log and step % 100 == 0:
            print("After %d steps, perplexity is %.3f" % (step, np.exp(total_costs / iters)))
    return np.exp(total_costs / iters)

#4. 定义主函数并执行。

def main():
    train_data, valid_data, test_data, _ = reader_ptb.ptb_raw_data(DATA_PATH)

    # 计算一个epoch需要训练的次数
    train_data_len = len(train_data)
    train_batch_len = train_data_len // TRAIN_BATCH_SIZE
    train_epoch_size = (train_batch_len - 1) // TRAIN_NUM_STEP

    valid_data_len = len(valid_data)
    valid_batch_len = valid_data_len // EVAL_BATCH_SIZE
    valid_epoch_size = (valid_batch_len - 1) // EVAL_NUM_STEP

    test_data_len = len(test_data)
    test_batch_len = test_data_len // EVAL_BATCH_SIZE
    test_epoch_size = (test_batch_len - 1) // EVAL_NUM_STEP

    initializer = tf.random_uniform_initializer(-0.05, 0.05)
    with tf.variable_scope("language_model", reuse=None, initializer=initializer):
        train_model = PTBModel(True, TRAIN_BATCH_SIZE, TRAIN_NUM_STEP)

    with tf.variable_scope("language_model", reuse=True, initializer=initializer):
        eval_model = PTBModel(False, EVAL_BATCH_SIZE, EVAL_NUM_STEP)

    # 训练模型。
    with tf.Session() as session:
        tf.global_variables_initializer().run()

        train_queue = reader_ptb.ptb_producer(train_data, train_model.batch_size, train_model.num_steps)
        eval_queue = reader_ptb.ptb_producer(valid_data, eval_model.batch_size, eval_model.num_steps)
        test_queue = reader_ptb.ptb_producer(test_data, eval_model.batch_size, eval_model.num_steps)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=session, coord=coord)

        for i in range(NUM_EPOCH):
            print("In iteration: %d" % (i + 1))
            run_epoch(session, train_model, train_queue, train_model.train_op, True, train_epoch_size)

            valid_perplexity = run_epoch(session, eval_model, eval_queue, tf.no_op(), False, valid_epoch_size)
            print("Epoch: %d Validation Perplexity: %.3f" % (i + 1, valid_perplexity))

        test_perplexity = run_epoch(session, eval_model, test_queue, tf.no_op(), False, test_epoch_size)
        print("Test Perplexity: %.3f" % test_perplexity)

        coord.request_stop()
        coord.join(threads)

if __name__ == "__main__":
    main()
