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
from tensorflow.python.framework import  graph_util
v1 = tf.Variable(initial_value=tf.constant(value=1.0,dtype=tf.float32,shape=[1]),name="v1")
v2 = tf.Variable(initial_value=tf.constant(value=2.0,dtype=tf.float32,shape=[1]),name="v2")

result =  v1 + v2
# 1. pb文件的保存方法。
init_op = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init_op)
    #导出当前计算图的GraphDef部分，只需要这一部分就可以完成从输入层到输出层的计算过程。
    grap_def = tf.get_default_graph().as_graph_def()

    #将图中的变量及其取值化为常量，同时将图中不必要的节点去掉。如果只关心程序中定义的某些计算时，和这些计算无关的节点就没有必要
    #导出并保存了。
    #下面一行代码，给出的add是计算节点的名称，所有后面没有0。
    #注释：张量名称后面有：0,表示某个计算节点的第一个输出。而计算节点本身后面没有:0的。
    output_graph_def = graph_util.convert_variables_to_constants(sess,grap_def,['add'])
    #将导出的模型存入文件
    with tf.gfile.GFile("model/combine_model.pb","wb") as f:
        f.write(output_graph_def.SerializeToString())

#2. 加载pb文件
from tensorflow.python.platform import gfile

with tf.Session() as sess:
    model_filename = "model/combine_model.pb"
    #读取保存的模型文件，并解析成GraphDef Protocol Buffer
    with gfile.FastGFile(model_filename,"rb") as f:
        grap_def = tf.GraphDef()
        grap_def.ParseFromString(f.read())
    #将graph_def 中保存的图加载到当前图中。return_elements={"add:0"} 给出返回的张量的名称。
    #在保存的时候给出的是计算节点的名称，所以为“add".在加载模型的时候给出的张量名称，所以是add:0
    result = tf.import_graph_def(grap_def,return_elements=["add:0"])
    print(sess.run(result))


