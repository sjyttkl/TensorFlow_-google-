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

# cross_entrop = - tf.reduce_mean(y_*tf.log(tf.clip_by_value(y,1e-10,10.0)))

v = tf.constant([[1.0,2.0,3.0],[4.0,5.0,6.0]])
v2 = tf.constant([1.0,2.0,3.0])
sess = tf.Session()
with sess.as_default():
    print( tf.clip_by_value(v,2.5,  4.5).eval(),tf.log(v2).eval())
v1 = tf.constant([[1.0,2.0],[3.0,4.0]])
v2 = tf.constant([[5.0,6.0],[7.0,8.0]])
#封装后的交叉熵
#cross_entropy  =tf.nn.softmax_cross_entropy_with_logits(v1,v2)
# tf.nn.sparse_softmax_cross_entropy_with_logits()#加速运行交叉熵
sess  = tf.Session()
with sess.as_default():
    print((v1*v2).eval())
    print(tf.matmul(v1,v2))
    print(tf.reduce_mean(v2).eval())
#分类问题，使用交叉熵为损失函数。
#回归问题可以使用MSE均方误差。 mean squared error
# mse = tf.reduce_mean(tf.square(y_-y))
#自定义损失函数，比如利润这个问题， 多预测一个，只是损失成本可能是1块钱，如果少预测一个则损失 1个利润可能是10块钱
a = 10
b =1
# tf.where()代替了 tf.select
# loss = tf.reduce_mean(tf.where(tf.greater(v1,v2),(v1-v2)*a,(v2-v1)*b)) #如果greater维度不一样，会进行广播操作

v1 = tf.constant([1.0,2.0,3.0,4.0])
v2 = tf.constant([4.0,3.0,2.0,1.0])

sess = tf.InteractiveSession()
print(tf.greater(v1,v2).eval())

print(tf.where(tf.greater(v1,v2),v1,v2).eval())
sess.close()

#指数衰减
# decayed_learning_rate = learning_rate*decay_rate^(global_step/decayed_steps)
global_step = tf.Variable(0)
#通过exponential——decay函数生成学习率
learning_rate = tf.train.exponential_decay(0.1,global_step,100,0.96,staircase=True)

#使用指数函数的学习率，在minimize函数中传入global_step将自动更新
# learning_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)
#=============================正则化====================
# w = tf.Variable(tf.random_normal([2,1],stddev=1,seed=1))
# y = tf.matmul(x,w)
# loss = tf.reduce_mean(tf.square(y_ - y))+tf.contrib.layers.12_regularizer(lambda )(w)
weights = tf.constant([[1.0,-2.0],[-3.0,4.0]])
with tf.Session() as sess:
    #输出为：（1 -2-3+4）*0.5
    print(sess.run(tf.contrib.layers.l1_regularizer(.5)(weights)))
    #输出为：(1^2+(-2)^2+(-3)^2+(4)^2) = 7.5 #Tensorflow 会把L2正则化除以2使得求导结果更加简洁
    print(sess.run(tf.contrib.layers.l2_regularizer(.5)(weights)))

sess=tf.InteractiveSession()
#初始化2个Variable
v1=tf.Variable(tf.constant(1))
v2=tf.Variable(tf.constant(1))
#设置保存到collection的name为collection
name='collection'
#把v1和v2添加到默认graph的collection中
tf.add_to_collection(name,v1)
tf.add_to_collection(name,v2)
#获得名为name的集合
c1 = tf.get_collection(name)
tf.global_variables_initializer().run()
print("the first collection: %s"%sess.run(c1))
#修改v1和v2的值,必须使用eval()或run()进行执行
tf.assign(v1,tf.constant(3)).eval()
tf.assign(v2,tf.constant(4)).eval()
#再次查看collection中的值
c2 = tf.get_collection(name)
print("the second collection: %s"%sess.run(c2))
print("the sum of collection: %s"%sess.run(tf.add_n(c2)))
#resut:
#the first collection: [1, 1]
#the second collection: [3, 4]
#the sum of collection: 7
