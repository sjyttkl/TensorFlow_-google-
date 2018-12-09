# -*- coding: UTF-8 -*-
"""
===============================================================
author：sjyttkl
email：695492835@qq.com
date：2018
introduction: 详细请见https://www.cnblogs.com/bmsl/p/dongbin_bmsl_01.html
===============================================================
"""

"""
    使用Tensorflow 实现卷积层，通过Tensorflow-slim可以在一行实现了一个卷积的向前传播算法。
    slim_conv2d的函数有三个参数是必填的。第一个参数为输入节点矩阵，第二个参数是当前卷积层过滤器的深度。
    第三个参数是过滤器的尺寸。可选的参数有过滤器移动的步长、是否全0填充、激活函数的选择一级变量的命名空间等
    net = slim.con2d(input,32,[3,3])
"""

__author__ = "sjyttkl"
import tensorflow.contrib.slim as slim
import tensorflow as tf


# slim.arg_scope()函数可以用于设置默认的参数取值。slim.arg_scope函数的第一个参数是一个列表。
#在这个列表中的函数将使用默认的参数取值。比如通过下面的定义，调用 slim.conv2d(net,320,[1,1])函数时会自动加上stride=1 和padding='SAME' 的参数
#如果在函数调用时候可以指定stride，那么这里设置的默认值就不会再使用。通过这种范式可以经一部减少冗余代码
with slim.arg_scope([slim.conv2d,slim.max_pool2d,slim.avg_pool2d],stride=1,padding='SAME'):
    '''
    此处省略了Inception-v3模型中其他的网络结构而直接实现最后面红色方框中的Inception结构。。
    假设输入图片经过之前的神经网络的向前传播的结构保存在变量net中
    '''
    net ="上一层的输出节点矩阵"


    with tf.variable_scope('Mixed_7c'):
        with tf.variable_scope("Branch_0"):
            branch_0 = slim.conv2d(net,320,[1,1],scope='Conv2d_0a_1x1')
        #Inception 模块中第二条路径。这计算路径上结构也是一个Inception结构。
        with tf.variable_scope(name_or_scope="Branch_1"):
            branch_1 = slim.conv2d(net,384,[1,1],scope='Conv2d_0a_1x1')
            #tf.concat函数可以将多个矩阵拼接起来。tf.concat函数的第一个参数指定了拼接维度，
            # 这里给出了“3”代表了矩阵是在深度这个维度上进行拼接。图6-16展示 深度上拼接矩阵的方式
            # 。
            branch_1 = tf.concat(values=3,axis=[ slim.conv2d(inputs=branch_1,num_outputs=384,kernel_size=[1,3],scope='Conv2d_0b_1x3'),
                slim.conv2d(branch_1, 384, [1, 3], scope='Conv2d_0b_3x1')])
        #Inception 模块中的第三条路径。这条计算路径上的机构本身也是一个Inception的结构
        with tf.variable_scope(name_or_scope="Branch_2"):
            branch_2 = slim.conv2d(net,448,[1,1],scope='Conv2d_0a_1x1')
            branch_2 = slim.conv2d(branch_2,384,[3,3],scope='Conv2d_0b_3x3')
            branch_2 = tf.concat(values=3,axis=[slim.conv2d(branch_2,384,[1,3],scope='Conv2d_0c_1x3'),
                                        slim.conv2d(branch_2,384,[3,1],scope='Conv2d_0d_3x1')])
        #Inception 模块中第四条路径
        with tf.variable_scope("Branch_3"):
            branch_3 = slim.avg_pool2d(net,[3,3],scope='AvgPool_0a_3x3')
            branch_3 = slim.conv2d(branch_3,192,[1,1],scope='Conv2d_0b_1x1')
        #当前Inception模块最后输出是由上面四个计算结果拼接得到的
        net = tf.concat(3,[branch_0,branch_1,branch_3,branch_3])




