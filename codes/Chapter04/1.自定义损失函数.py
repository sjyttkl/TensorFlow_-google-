# -*- coding: UTF-8 -*-
"""
===============================================================
author：sjyttkl
email：695492835@qq.com
date：2018
introduction:
===============================================================
"""

import tensorflow as tf
from numpy.random import RandomState

#1. 定义神经网络的相关参数和变量。
batch_size = 8
x = tf.placeholder(tf.float32,shape=(None,2),name="x-input")
#回归问题，一般只有一个输出节点
y_ = tf.placeholder(tf.float32,shape=(None,1),name="y-input")
#定义了一个单层的神经网络向前传播的过程，这里就是简单的加权和
w1 = tf.Variable(tf.random_normal([2,1],stddev=1,seed=1))
y = tf.matmul(x,w1)

#2. 设置自定义的损失函数。
#定义预测多了和预测少了的成本
## 定义损失函数使得预测少了的损失大，于是模型应该偏向多的方向预测。
# loss_less = 10
# loss_more = 1
#重新定义损失函数，使得预测多了的损失大，于是模型应该偏向少的方向预测。¶
loss_less = 1
loss_more = 10
# loss = tf.reduce_sum(tf.where(tf.greater(y,y_),(y-y_)*loss_more,(y_-y)*loss_less))
#6. 定义损失函数为MSE
loss = tf.contrib.losses.mean_squared_error(y, y_)
train_step = tf.train.AdamOptimizer(0.001).minimize(loss)
#通过随机数生成一个模拟数据集
rdm = RandomState(1)
dataset_size = 128
X = rdm.rand(dataset_size,2)
#设置回归的正确值为两个输入的和加上一个随机量，之所以要加上一个随机量是为了
#加入不可预测的噪音，否则不同损失函数的意义就不懂了，因为不同损失函数都会在能
#完全预测正确的时候最低。一般来说噪音为了一个均值为0的小量，所以这里噪音设置为
#-0.05 ~ 0.05 的随机数
Y = [[x1 +x2 +rdm.rand() /10.0-0.05 ] for (x1,x2) in X]

#训练神经网络
with tf.Session() as sess:
    init_op = tf.initialize_all_variables()
    sess.run(init_op)
    STEPS = 5000
    for i in range(STEPS):
        start = (i*batch_size) %dataset_size
        end = min(start+batch_size,dataset_size)
        sess.run(train_step,feed_dict={x:X[start:end],y_:Y[start:end]})
        if i % 1000 == 0:
            print("After %d training step(s), w1 is: " % (i))
            print(sess.run(w1), "\n")
    print("Final w1 is: \n", sess.run(w1))






























