# -*- coding: utf-8 -*-

"""
==================================================
   File Name：     1.命名空间
   email:         songdongdong@jd.com
   Author :       songdongdong
   date：          2019/3/31
   Description :  
==================================================
"""
__author__ = 'songdongdong'
# 1. 不同的命名空间。
import tensorflow as tf
with tf.variable_scope("foo"):
    a = tf.get_variable("bar",[1])
    print(a,a.name)
# 2. tf.Variable和tf.get_variable的区别。
with tf.name_scope("a"):
    a = tf.Variable([1])
    print(a.name)
    a = tf.get_variable("b",[1])
    print(a.name)
# 3. TensorBoard可以根据命名空间来整理可视化效果图上的节点。
with tf.name_scope("input1"):
    input1 = tf.constant([1.0,2.0,3.0],name="input2")
with tf.name_scope("input2"):
    input2 = tf.Variable(tf.random_normal([3]),name="input2")
output = tf.add_n([input1,input2],name="add")

writer = tf.summary.FileWriter("log/simple_example.log",tf.get_default_graph())
writer.close()