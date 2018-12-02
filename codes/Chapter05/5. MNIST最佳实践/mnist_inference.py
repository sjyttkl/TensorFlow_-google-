# -*- coding: UTF-8 -*-
"""
===============================================================
author：sjyttkl
email：695492835@qq.com
date：2018
introduction:这段代码定义了向前传播的算法，无论是训练时还是在测试时候，都可以直接调用inference这个函数，而不需要关系具体的网络结构
===============================================================
"""
__author__ = "sjyttkl"
import tensorflow as tf
INPUT_NODE = 784
OUTPUT_NODE = 10
LAYER1_NODE = 500

# 2. 通过tf.get_variable函数来获取变量
def get_weight_variable(shape,regularizer):
    weights = tf.get_variable(name="weights",shape=shape,initializer=tf.truncated_normal_initializer(stddev=0.1))
    if regularizer != None:tf.add_to_collection("losses",regularizer(weights))
    return weights

# 3. 定义神经网络的前向传播过程。
def inference(input_tensor,regularizer):
    with tf.variable_scope("layer1"):
        weights = get_weight_variable([INPUT_NODE,LAYER1_NODE],regularizer)
        biases = tf.get_variable('biases',[LAYER1_NODE],initializer=tf.constant_initializer(0.0))
        layer1 = tf.nn.relu(tf.matmul(input_tensor,weights)+biases)
    with tf.variable_scope("layer2"):
        weights = get_weight_variable([LAYER1_NODE, OUTPUT_NODE], regularizer)
        biases = tf.get_variable('biases', [OUTPUT_NODE], initializer=tf.constant_initializer(0.0))
        layer2 = tf.matmul(layer1, weights) + biases
    return layer2
