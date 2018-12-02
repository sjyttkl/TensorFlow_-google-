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

import tensorflow as tf

# 1. 使用滑动平均。
# v = tf.Variable(0,dtype=tf.float32,name="v")
# #在没有声明滑动平均模型时只有一个变量v,所以下面的语句只会输出：v:0
# for variables in tf.global_variables():
#     print(variables.name)
# ema = tf.train.ExponentialMovingAverage(0.99)
# maintain_averages_op  = ema.apply(tf.global_variables())
# #在申明滑动平均模型之后，Tensorflow会自动生成一个影子变量
# for variables in tf.global_variables():
#     print(variables.name)
# #v:0
# # v/ExponentialMovingAverage:0
#
# saver = tf.train.Saver()
# with tf.Session() as sess:
#     inio_op = tf.initialize_all_variables()
#     sess.run(inio_op)
#
#     sess.run(tf.assign(v,10))
#     sess.run(maintain_averages_op)
#     #保存Tensorflow会将v:0 和 v/ExponentialMovingAverage:0 两个变量都保存下来了
#     saver.save(sess,"model/aver_model")
#     print(sess.run([v,ema.average(v)]))

# 3. 加载滑动平均模型。
v = tf.Variable(initial_value=0,dtype=tf.float32,name="v")
#通过变量重命名将原来的滑动平均值直接赋值给v
saver = tf.train.Saver({"v/ExponentialMovingAverage":v})
with tf.Session() as sess:
    saver.restore(sess,"model/aver_model")
    print(sess.run(v))
#
