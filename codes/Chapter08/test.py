# -*- coding: utf-8 -*-

"""
==================================================
   File Name：     test.py
   email:         songdongdong@jd.com
   Author :       songdongdong
   date：          2019/3/26
   Description :  
==================================================
"""
__author__ = 'songdongdong'

import tensorflow as tf
a = tf.constant([1.0, 2.0], name="a")
b = tf.constant([2.0, 3.0], name="b")
result = a + b
print (result)


sess = tf.InteractiveSession ()
print(result.eval())
sess.close()

sess = tf.Session()
with sess.as_default():
    print(result.eval)
    # print(tf.shape(result, [-1]))

print(sess.run(result))
print(result.eval(session=sess))
