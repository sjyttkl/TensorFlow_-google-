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
# 1. 在上下文管理器“foo”中创建变量“v”。
with tf.variable_scope("foo"):
    v = tf.get_variable("v",shape=[1],initializer=tf.constant_initializer(1.0))
# with tf.variable_scope("foo"):
#     v = tf.get_variable("v",shape=[1])  #同一个foo下，不可以重复创建
with tf.variable_scope("foo",reuse = True):
    v1 = tf.get_variable("v",shape=[1])
# print(v==v1)
# print(v,v1)
# #2. 嵌套上下文管理器中的reuse参数的使用
# with tf.variable_scope("root"):
#     print(tf.get_variable_scope().reuse)
#     with tf.variable_scope("foo",reuse=True):
#         print(tf.get_variable_scope().reuse)
#         with tf.variable_scope("bar"):
#             print(tf.get_variable_scope().reuse)
#     print(tf.get_variable_scope().reuse)

#3. 通过variable_scope来管理变量。
v = tf.get_variable("v",[1])
vv = tf.Variable("v")
vv3 = tf.Variable("v")
print(v.name,vv.name,vv3.name)
with tf.variable_scope("foo",reuse=True):
    v2 = tf.get_variable("v",[1])
print(v2.name)
with tf.variable_scope("foo"):
    with tf.variable_scope("bar"):
        v3 = tf.get_variable("v",[2])
        print(v3.name)
    v4 = tf.get_variable("v1",[3])
    print(v4.name)

#4. 我们可以通过变量的名称来获取变量
with tf.variable_scope("",reuse=True):
    v5 = tf.get_variable("foo/bar/v",[2])
    print(v5 == v3)
    v6 = tf.get_variable('foo/v1',[3])
    print(v6==v4)

