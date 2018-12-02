# -*- coding: UTF-8 -*-
"""
===============================================================
author：sjyttkl
email：695492835@qq.com
date：2018
introduction:
===============================================================
"""
__author__ = "sjyttkl"
import  tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 1.设置输入和输出节点的个数,配置神经网络的参数。
INPUT_NODE = 784     # 输入节点
OUTPUT_NODE = 10     # 输出节点
LAYER1_NODE = 500    # 隐藏层数

BATCH_SIZE = 100 # 每次batch打包的样本个数
#模型的相关参数
LEARNING_RATE_BASE = 0.8  #学习率
LEARNING_RATE_DECAY = 0.99   # 学习率的衰减率
REGULARAZTION_RATE = 0.0001 # 正则化项在损失函数中的系数
TRAINING_STEPS = 5000       #训练步数
MOVING_AVERAGE_DECAY = 0.99 #滑动平均衰减率  ,越大就约稳定。

#一个辅助函数，给定神经网络的输入和所有参数，计算神经网络的前向传播结果。
#在这里定义了一个relu激活函数的三层全连接神经网络。通过加入隐藏层实现了多层网络结构，
#通过ReLu激活函数实现去线性化，在这个函数中也支持传入用于计算参数平均值的类。
#这样方便的测试时使用滑动平均模型---滑动平均值一般在测试的时候使用
def inference(input_tensor,avg_class,weights1,biases1,weights2,biases2):
    #当没有滑动平均类时，直接使用参数当前的取值
    if avg_class == None:
        #计算隐藏层的向前传播的结果，这里使用ReLU激活函数
        layer1 = tf.nn.relu(tf.matmul(input_tensor,weights1)+biases1)
        #计算输出层的向前的传播结果，因为在计算损失函数时会一并使用softmax函数，
        #所以这里不需要加入激活函数。而且不加入softmax不会影响预测结果。因为预测时候
        #使用的是不同类别对应节点输出值的相对大小，有没有softmax层对最后的分类结果的计算没有影响。
        #于是在就散整个神经网络的前向传播时可以不加入最后的softmax结果。
        return tf.matmul(layer1,weights2)+biases2
    else:
        #使用avg_class.average函数来计算滑动平均值。
        #然后再计算相应的神经网络向前传播结果
        layer1 = tf.nn.relu(tf.matmul(input_tensor,avg_class.average(weights1))+ avg_class.average(biases1))
        return tf.matmul(layer1,avg_class.average(weights2))+ avg_class.average(biases2)
#3. 定义训练过程。
def trian(mnist):
    x = tf.placeholder(name="x_input",dtype=tf.float32,shape=[None,INPUT_NODE])
    y_ = tf.placeholder(name="y_input",dtype=tf.float32,shape=[None,OUTPUT_NODE])
    # 生成隐藏层的参数。
    weights1 = tf.Variable(name="weights1",initial_value=tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev=0.1))
    biases1 = tf.Variable(name="biases1",initial_value=tf.constant(0.1, shape=[LAYER1_NODE]))
    # 生成输出层的参数。
    weights2 = tf.Variable(name="weights2",initial_value=tf.truncated_normal([LAYER1_NODE, OUTPUT_NODE], stddev=0.1))
    biases2 = tf.Variable(tf.constant(value=0.1, shape=[OUTPUT_NODE]))

    #计算不含滑动平均类的向前传播算法结果
    y = inference(input_tensor=x,avg_class=None,weights1=weights1,biases1=biases1,weights2=weights2,biases2=biases2)
    # 定义训练轮数及相关的滑动平均类
    global_step = tf.Variable(0,trainable=False)#一般会将代表训练轮数的变量指定为不可训练的参数
    #计算滑动平均数，给定训练轮数的变量，可以加快训练早期变量的更新速度
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,num_updates=global_step)
    #在所有代表网络参数的变量上使用滑动平均值，其他辅助变量(global_setp)就不需要了),，tf.train_variables(）返回的是图上的集合
    #Graphkeys.TRAINABLE_VARIABLES中的元素，这个集合的元素没有指定，trianable=false的参数
    variables_averages_op = variable_averages.apply(tf.trainable_variables())#把需要迭代，全都传进去吧
    average_y = inference(input_tensor=x,avg_class=variable_averages,weights1=weights1,biases1=biases1,weights2=weights2,biases2=biases2)

    #计算交叉熵及其平均值，在问题只有一个答案的时候，可以用这个函数来加速交叉熵的计算。第一个需要传入的值是需要不经过sofmax函数的值
    cross_entorpy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=tf.argmax(y_,1))
    #在所有的batch中都需要计算交叉熵平均值
    cross_entropy_mean = tf.reduce_mean(cross_entorpy)

    #损失函数计算
    regularizer =  tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)
    regularation  = regularizer(weights1)+regularizer(weights2)
    loss = cross_entropy_mean +regularation

    ## 设置指数衰减的学习率。
    learning_rate = tf.train.exponential_decay(learning_rate=LEARNING_RATE_BASE,#基础学习率，
                                               global_step=global_step,#当前迭代次数
                                               decay_steps=mnist.train.num_examples/BATCH_SIZE,#过完所有的训练，需要迭代的次数
                                               decay_rate=LEARNING_RATE_DECAY,#学习衰减速度
                                               staircase=True)#为ture则为离散区间的学习率
    #优化损失函数
    trian_step = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss,global_step=global_step)
    #反向传播函数更新参数和更新每一个参数的滑动平均值,需要同时进行，下面两种方式都可以，
    # train_op = tf.group([trian_step,variables_averages_op])
    with tf.control_dependencies([trian_step,variables_averages_op]):
        train_op = tf.no_op(name="train")#tf.no_op；什么也不做，确保a，b按顺序都执行。
        #https://blog.csdn.net/PKU_Jade/article/details/73498753 这里讲的很详细。
    #就算正确率
    corrrect_prediction = tf.equal(tf.argmax(average_y,1),tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(corrrect_prediction,tf.float32))
    # 初始化回话并开始训练过程。
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        #验证数据，一般会用验证数据来大致判断停止的条件和评判训练的效果
        vaildate_feed = {x:mnist.validation.images,
                         y_:mnist.validation.labels}
        #准备测试数据。在真实应用中，这部分数据在训练时是不可见的。这个数据只是作为模型优劣的最后评价标准
        test_feed = {x:mnist.test.images,
                     y_:mnist.test.labels}

        #迭代的训练神经网络
        for i in range(TRAINING_STEPS):
            #每100轮输出一次在验证数据集上的测试结果
            # sess.run(tf.assign(global_step, 5))
            if i %1000 == 0:
                validate_acc = sess.run(accuracy,feed_dict=vaildate_feed)
                print("After %d training step(s), validation accuracy using average model is %g " % (i, validate_acc))

            xs ,ys = mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op,feed_dict={x:xs,y_:ys})
        test_acc = sess.run(accuracy,feed_dict=test_feed)
        print(("After %d training step(s), test accuracy using average model is %g" % (TRAINING_STEPS, test_acc)))
def main(argv=None):
    mnist = input_data.read_data_sets("../datasets/MNIST_data", one_hot=True)
    trian(mnist)

if __name__ =="__main__":
    main()










