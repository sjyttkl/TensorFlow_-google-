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
#1. 设定神经网络的参数
INPUT_NODE = 784
OUTPUT_NODE = 10

IMAGE_SIZE = 28
NUM_CHANNELS = 1
NUM_LABELS = 10

#第一层卷积的尺寸和深度
CONV1_DEEP=32
CONV1_SIZE =5
#第二层卷积层的尺寸和深度
CONV2_DEEP = 64
CONV2_SIZE=5

#全连接层的节点个数。
FC_SIZE = 512

# 2. 定义前向传播的过程

#定义卷积神经网络的向前传播过程，添加了一个新的参数train，用来区分训练过程和测试过程。在这个程序中将用到dropout方法，
#dropout可以进一步提升模型可靠性并防止过拟合，dropout过程只是在训练的时候使用
def inference(input_tensor,train,regularizer):
    with tf.variable_scope(name_or_scope="layer1-conv1"):
        conv1_weights = tf.get_variable(name="weight",shape=[CONV1_SIZE,CONV1_SIZE,NUM_CHANNELS,CONV1_DEEP],initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases = tf.get_variable(name="bias",shape=[CONV1_DEEP],initializer=tf.constant_initializer(value=0.0))

        #使用边长为5，深度为32的过滤器，过滤器移动的步长为 1， 且全用0填充。
        conv1 = tf.nn.conv2d(input=input_tensor,filter=conv1_weights,strides=[1,1,1,1],padding='SAME')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1,conv1_biases))
    #实现第二次池化层的向前传播过，这里选用最大池化层，池化层的过滤器的边长为2，使用全0填充且移动的步长为2。
    #这一层的输入时上一层的疏忽，也就是28*28*32的矩阵。输出为14*14*32的矩阵
    with tf.name_scope('layer2-pool1'):
        pool1 = tf.nn.max_pool(value=relu1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    #声明第三层卷积层的变量并实现向前传播的过程，这一层的输入为14*14*32的矩阵
    #输出为14*14*64的矩阵
    with tf.variable_scope('layer3-conv2'):
        conv2_weights = tf.get_variable(name='weight',shape=[CONV2_SIZE,CONV2_SIZE,CONV1_DEEP,CONV2_DEEP],initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable(name='bias',shape=[CONV2_DEEP],initializer=tf.constant_initializer(0.0))
        #使用边长为5，深度64的过滤器，过滤器的移动步长为1，且使用全为0 填充
        conv2 = tf.nn.conv2d(input=pool1,filter=conv2_weights,strides=[1,1,1,1],padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2,conv2_biases))
        # 实现第四层的向前传播过程，这一层和第二的结构式一样的。这一层的输入为14*14*64，输出Wie7*7*64的矩阵
    with tf.name_scope('layer4-pool2'):
        pool2 = tf.nn.max_pool(relu2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
#将第四层池化层输出为第五层全连接的输入格式。在第四层的输出为7*7*64的矩阵。然而这第五层的全连接层需要输入格式为向量。所以这里需要将这个7*7*64的矩阵拉
#直成一个向量。pool2.get_shape函数可以第四层输出的矩阵的维度，而不需要手工计算。注意：因为每一层神经网络的输入和输出都为一个batch的矩阵，
#所以这里得到的维度也包含了一个batch中数据的个数,
    pool_shape = pool2.get_shape().as_list()
    #计算将矩阵拉直成为向量之后的长度，这个长度就是矩阵长宽及深度的乘积，注意这里pool_shape[0] 为一个batch 中数据的个数
    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
    #通过tf.reshape函数将第四层的输出变成一个batch的向量
    reshaped = tf.reshape(pool2,[pool_shape[0],nodes])

    #声明第五层全连接层的变量并实现向前传播的过程。这一层的输入时拉直之后的一组向量，向量长度为3136，输出是一组长度为512的向量
    #dropout一般只是在全连接层而不是卷积层或者池化层使用
    with tf.variable_scope(name_or_scope="layer5-fc1"):
        fc1_weights = tf.get_variable(name="weight",shape=[nodes,FC_SIZE],initializer=tf.truncated_normal_initializer(stddev=0.1))
        #只有全连接层的权重需要加入正则化
        if regularizer != None:
            tf.add_to_collection(name="lossess",value=regularizer(fc1_weights))

        fc1_biases = tf.get_variable(name="bias",shape=[FC_SIZE],initializer=tf.constant_initializer(0.1))
        fc1 = tf.nn.relu(tf.matmul(a=reshaped,b=fc1_weights)+fc1_biases)
        if train : fc1 = tf.nn.dropout(x=fc1,keep_prob=0.5)

    #判断第六层全连接层的变量并实现向前传播的过程，这一层的输入为一组长度为512的向量，
    #输出为一组长度为10的向量。这一层的输出通过Softmax之后就得到了最后的分类结果
    with tf.variable_scope('layer6-fc2'):
        fc2_weights = tf.get_variable(name="weight",shape=[FC_SIZE,NUM_LABELS],initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None:
            tf.add_to_collection(name="losses",value=regularizer(fc2_weights))
        fc2_biases = tf.get_variable(name='bias',shape=[NUM_LABELS],initializer=tf.truncated_normal_initializer(0.1))
        logit = tf.matmul(a=fc1,b=fc2_weights) +fc2_biases
    #返回第六次层的输出
    return logit







