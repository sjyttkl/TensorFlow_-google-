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

v1 = tf.Variable(tf.constant(value=1.0,shape=[1]),name="v1")
v2 = tf.Variable(tf.constant(value=2.0,shape=[1]),name="v2")
result1 = v1 + v2

saver = tf.train.Saver()
#通过export_meta_graph函数导出Tensorflow计算图的元图，并保存为json格式
saver.export_meta_graph("model/model.ckpt.meta.json",as_text=True)

#model.ckpt问中列表的第一行描述文件的元信息，比如这个文件中存储的变量列表。
#列表下每一行报错了一个变量的片段，变量片段的信息通过SavedSlice ProtocolBuffer定义的。
#SaverSlice类型中保存了变量的名称、单曲片段的信息以及变量的取值。
#Tensorflow提供了tf.train.NewCheckpointReader类来查看model.ckpt文件中保存的变量信息
#tf.train.NewCheckpointReader可以读取checkpoint文件中保存的所有变量
reader = tf.train.NewCheckpointReader("model/model.ckpt")
#获取所有变量的列表，这个是从变量名到变量维度的字典
all_variables = reader.get_variable_to_shape_map()
for variable_name in all_variables:
    print(variable_name,all_variables[variable_name])
#获取名称为v1的变量取值
print("Value for variable v1 is ",reader.get_tensor("Variable_1"))
