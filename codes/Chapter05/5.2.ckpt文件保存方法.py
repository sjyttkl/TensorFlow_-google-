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

import tensorflow as tf

# 1. 保存计算两个变量和的模型
#声明两个变量并计算它们的和
v1 = tf.Variable(tf.constant(1.0,shape=[1],name="v1"))
v2 = tf.Variable(tf.constant(2.0,shape=[1],name="v2"))
result =  v1 + v2


#声明tf.train.Saver()保存模型
saver = tf.train.Saver()
# init_op = tf.initialize_all_variables()
# with tf.Session() as sess:
#     sess.run(init_op)
#     saver.save(sess,"model/model.ckpt")

# # 加载保存了两个变量和的模型
with tf.Session() as sess:
    saver.restore(sess, "model/model.ckpt")
    print (sess.run(result))

#直接加载持久化的图
saver = tf.train.import_meta_graph("model/model.ckpt.meta")
with tf.Session() as sess:
    saver.restore(sess,"model/model.ckpt")
    #通过张量名称来获取张量
    print(sess.run(tf.get_default_graph().get_tensor_by_name("add:0")))


# 变量重命名。·
#这里声明的变量名称和已经保存的模型中变量名称不同
v1 = tf.Variable(tf.constant(1.0,shape=[1],name="v1-other"))
v2 = tf.Variable(tf.constant(2.0,shape=[1],name="v2-other"))

#如果直接使用tf.train.Saver() 来加载模型会报变量找不到的错误，
# 使用一个字典来重命名变量可以就可以加载原来的模型了，这个字典指定了原来名称
# v1的变量，现在加载到变量v1中（名称为v1-other),名称v2的变量加载到变量v2中（名称为v2-other)。
saver = tf.train.Saver({"v1":v1,"v2":v2})
# 这样做的原因是方便使用滑动平均值，


#变量重命名。

