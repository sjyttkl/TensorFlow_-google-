# -*- coding: UTF-8 -*-
"""
===============================================================
author：songdongdong
email：songdongdong@jd.com
date：2018
introduction:
===============================================================
"""
#1，定义两个不同的图
import tensorflow as tf
g1 = tf.Graph()
with g1.as_default():

    v = tf.get_variable("v",[1],initializer=tf.zeros_initializer())# 设置初始值为0

g2 = tf.Graph()
with g2.as_default():
    v = tf.get_variable("v",[1],initializer=tf.ones_initializer())# 设置初始值为1


with tf.Session(graph=g1) as sess:
    tf.global_variables_initializer().run()
    with tf.variable_scope("",reuse=True):
        print(sess.run(tf.get_variable("v")))

with tf.Session(graph=g2) as sess:
    tf.global_variables_initializer().run()
    with tf.variable_scope("",reuse=True):
        print(sess.run(tf.get_variable("v")))

print("----张量的概念------")
# 2. 张量的概念
import tensorflow as tf
a = tf.constant([1.0, 2.0], name="a")
b = tf.constant([2.0, 3.0], name="b")
result = a + b
print (result)

print(v.graph is tf.get_default_graph())

sess = tf.InteractiveSession ()
print(result.eval())
sess.close()

sess = tf.Session()
with sess.as_default():
    print(result.eval)

print(sess.run(result))
print(result.eval(session=sess))
