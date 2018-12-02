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

#1.1定义变量
w1 = tf.Variable(tf.random_normal([2,3],stddev=1,seed=1))#stddev样本标准偏差
w2 = tf.Variable(tf.random_normal([3,1],stddev=1,seed=1))#stddev样本标准偏差
x = tf.constant([[0.7, 0.9]])
#1.2 定义向前传播的神经网络

a = tf.matmul(x,w1)
y = tf.matmul(a,w2)

# 1.3 调用会话输出结果
sess = tf.Session()
sess.run(w1.initializer)
sess.run(w2.initializer)
print(sess.run(y))
sess.close()

#2. 使用placeholder

x=  tf.placeholder(tf.float32,shape=(1,2),name="input")
a = tf.matmul(x,w1)
y = tf.matmul(a,w2)

sess = tf.Session()

init_op = tf.global_variables_initializer()
sess.run(init_op)
print(sess.run(y,feed_dict={x:[[0.7,0.9]]}))

#3. 增加多个输入

x = tf.placeholder(tf.float32, shape=(3, 2), name="input")
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

sess = tf.Session()
#使用tf.global_variables_initializer()来初始化所有的变量
init_op = tf.global_variables_initializer()
sess.run(init_op)

print(sess.run(y, feed_dict={x: [[0.7,0.9],[0.1,0.4],[0.5,0.8]]}))









