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

#滑动平均模型：
#可以在测试集上的效果变得更健壮。
#shadow_variable = decay * shadow_variable +(1 - decay)*variable
#deacy 为衰减率，deacy越大，越趋于稳定，variable待衰减的参数，shadow_variale为影子参数
#tf.ExponentialMovingAverage(0.99,um_updates)，可以动态的设置deacy的大小

#1. 定义变量及滑动平均类
import tensorflow as tf
v1 = tf.Variable(0, dtype=tf.float32)
v2 = tf.Variable(1,dtype=tf.float32)
step = tf.Variable(0, trainable=False)#手动指定迭代次数，可以用于动态控制衰减率
#定义一个滑动平均类。初始化衰减率0.99和控制衰减率：step
ema = tf.train.ExponentialMovingAverage(0.99, step)
#定义一个更新变量滑动平均的操作，这里需要一个列表，每次执行这个操作。
#这个列表里的变量都会被更新
maintain_averages_op = ema.apply([v1,v2]) #这个方法会执行正在计算滑动平均的操作，还有记录效果,如果需要计算衰减的，就直接传过去吧

#2. 查看不同迭代中变量取值的变化。
with tf.Session() as sess:
    # 初始化
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    print(sess.run([v1, ema.average(v1)]))#第一次，都为0

    # 更新变量v1的取值
    sess.run(tf.assign(v1, 5))
    #更新v1的滑动平均值，衰减率为：min{0.99,(1+step)/(10+step)=0.1} = 0.1，这个公式，是为了前期会更新的比较快。
    #所以v1的滑动平均值会被更新为0.1 * 0 + 0.9*5 = 4.5
    print(sess.run(maintain_averages_op)) #没有返回值
    print(sess.run([v1, ema.average(v1)]))

    # 更新step和v1的取值
    sess.run(tf.assign(step, 10000))
    sess.run(tf.assign(v1, 10))
    # 更新v1的滑动平均值，衰减率为：min{0.99,(1+step)/(10+step)=0.9999} = 0.999
    # 所以v1的滑动平均值会被更新为0.99 * 4.5 + 0.001*10 = 4.55555
    sess.run(maintain_averages_op)
    print(sess.run([v1, ema.average(v1)]))

    # 更新一次v1的滑动平均值
    # 再次更新滑动平均值为0.99 *4.555 +0.01 *10 = 4.60945
    sess.run(maintain_averages_op)
    print(sess.run([v1, ema.average(v1)]))
