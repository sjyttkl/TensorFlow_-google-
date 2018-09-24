# -*- coding: UTF-8 -*-
"""
===============================================================
author：songdongdong
email：songdongdong@jd.com
date：2018.09.24
introduction:假设我们要最小化函数  y=x^2, 选择初始点   x_0=5
===============================================================
"""
__author__ = "songdongdong"

import tensorflow as tf
#1. 学习率为1的时候，x在5和-5之间震荡。
TRAINING_STEPS = 10
LEARNING_RATE=1
x  = tf.Variable(tf.constant(5,dtype=tf.float32),name="x")
y = tf.square(x)
trian_op = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(y)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(TRAINING_STEPS):
        sess.run(trian_op)
        x_value = sess.run(x)
        print("After %s iteration(s): x%s is %f."% (i+1, i+1, x_value))

 # 学习率为0.001的时候，下降速度过慢，在901轮时才收敛到0.823355。
TRAINING_STEPS = 1000
LEARNING_RATE = 0.001
x = tf.Variable(tf.constant(5, dtype=tf.float32), name="x")
y = tf.square(x)
trian_op = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(y)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(TRAINING_STEPS):
        sess.run(trian_op)
        if i % 100 == 0:
            x_value = sess.run(x)
            print("After %s iteration(s): x%s is %f." % (i + 1, i + 1, x_value))

# 3. 使用指数衰减的学习率，在迭代初期得到较高的下降速度，可以在较小的训练轮数下取得不错的收敛程度。
# decayed_learning_rate = learning_rate*decay_rate^(global_step/decayed_steps)
# 公式中，learning_rate： 当前的学习速率
# start_rate：最初的学习速率
# decay_rate：每轮学习的衰减率，0<decay_rate<1
# global_step：当前的学习步数，等同于我们将 batch 放入学习器的次数 ,它从学习速率第一次训练开始变化，global_steps每次自动加1
# decay_steps：每轮学习的步数，decay_steps = sample_size/batch  即样本总数除以每个batch的大小
TRAINING_STEPS = 100
global_step = tf.Variable(0)
LEARNING_RATE = tf.train.exponential_decay(0.1, global_step, 1, 0.96, staircase=True)

x = tf.Variable(tf.constant(5, dtype=tf.float32), name="x")
y = tf.square(x)
train_op = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(y, global_step=global_step)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(TRAINING_STEPS):
        sess.run(train_op)
        if i % 10 == 0:
            LEARNING_RATE_value = sess.run(LEARNING_RATE)
            x_value = sess.run(x)
            print("After %s iteration(s): x%s is %f, learning rate is %f."% (i+1, i+1, x_value, LEARNING_RATE_value))