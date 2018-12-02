# -*- coding: UTF-8 -*-
"""
===============================================================
author：songdongdong
email：songdongdong@jd.com
date：2018
introduction:
===============================================================
"""
__author__ = "songdongdong"
#w为了方便加载时重命名滑动平均变量，#tf.train.ExponentialMovingAverage类提供了variables_to_restore函数来生成tf.train.Saver类
#所需要的变量重名命字典。
import tensorflow as tf
v = tf.Variable(initial_value=0,dtype=tf.float32,name="v")
ema = tf.train.ExponentialMovingAverage(decay=0.99)
#通过使用variables_to_restore 函数可以直接生成上面代码中提供的字典
# {"v/ExponentialMovingAverage":v}
