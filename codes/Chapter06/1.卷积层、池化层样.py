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
import numpy as np
# 1. 假设我们输入矩阵
M = np.array([
        [[1],[-1],[0]],
        [[-1],[2],[1]],
        [[0],[2],[-2]]
    ])
print(M.shape)
#定义卷积过滤器深度为1
filter_weight = tf.get_variable(name="weigths",shape=[2,2,1,1],initializer=tf.constant_initializer([[1,-1],[0,2]]))
biases = tf.get_variable(name="biases",shape=[1],initializer=tf.constant_initializer(1))
# 3.调整输入的格式符合TensorFlow的要求。
M = np.array(p_object=M,dtype="float32")
M = M.reshape(1,3,3,1)
#4, 计算矩阵通过卷积层过滤器和池化层过滤器计算后的结果。

# x 的 形式为[batch, in_height, in_width, in_channels]`
x= tf.placeholder(dtype="float32",shape=[1,None,None,1])
# x input tensor of shape `[batch, in_height, in_width, in_channels]`
# W filter / kernel tensor of shape [filter_height, filter_width, in_channels, out_channels]
# `strides[0] = strides[3] = 1`. strides[1]代表x方向的步长，strides[2]代表y方向的步长
# padding: A `string` from: `"SAME", "VALID"`
conv = tf.nn.conv2d(x,filter_weight,strides=[1,2,2,1],padding="SAME")
bias = tf.nn.bias_add(conv,biases)
pool =tf.nn.avg_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    convoluted_M = sess.run(bias,feed_dict={x:M})
    pooled_M = sess.run(pool,feed_dict={x:M})

    print("convoluted_M: \n", convoluted_M)
    print("pooled_M: \n", pooled_M)

