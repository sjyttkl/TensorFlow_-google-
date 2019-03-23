# -*- coding: UTF-8 -*-
"""
===============================================================
author：sjyttkl
email：695492835@qq.com
date：2018
introduction:  详细说明请看 说明.txt  文档

这个模型，组成是这样的：

1）使用一个最简单的单层的神经网络进行学习

2）用SoftMax来做为激活函数

3）用交叉熵来做损失函数

4）用梯度下降来做优化方式
===============================================================
"""

import pandas as pd
import numpy as np
import tensorflow as tf
#1加载数据集，把对输入和结果和进行分开
train = pd.read_csv("train.csv")
images = train.iloc[:,1:].values
labels_flat = train.iloc[:,0].values.ravel()#这里是把二维变成一维注意区别numpy.ravel() vs numpy.flatten()

# 2对输入进行处理。
images = images.astype(np.float)
images = np.multiply(images,1.0/255.0) #我猜是归一化。
print("输入数据的量 {%g , %g}" %images.shape)

image_size = images.shape[1]
print("输入数据的维度  => {0}".format(image_size))

#这里是sqrt开方得到的 宽和高。因为之前的是把图像二维数据按照一维数据存进去的
image_width = image_height = np.ceil(np.sqrt(image_size).astype(np.uint8))
print("图片的长度=> {0} \n 图片的高 => {1}".format(image_width,image_height))

x = tf.placeholder('float',shape=[None,image_size])



#3对结果进行处理,unique去重
labels_count = np.unique(labels_flat).shape[0]
print("结果的种类 => {0}".format(labels_count))

y = tf.placeholder('float', shape=[None, labels_count])

#进行One-Hot编码 #另外一种方式：tf.one_hot（label，10）
def dense_to_one_hot(labels_dense,num_classes):
    num_labels= labels_dense.shape[0]
    index_offset = np.arange(num_labels)*num_classes
    labels_one_hot = np.zeros((num_labels,num_classes))
    x = index_offset+labels_dense.ravel()
    labels_one_hot.flat[index_offset+labels_dense.ravel()] = 1
    return labels_one_hot

labels = dense_to_one_hot(labels_flat,labels_count)
labels = labels.astype(np.uint8)
print("结果的数量：（{0[0]} , {0[1]}）".format(labels.shape))



#4把输入结果分为训练集和测试集
#把4000个数据做为训练集，2000个数据作为验证集
VALIDATION_SIZE = 2000

validation_images = images[:VALIDATION_SIZE]
validation_labels = labels[:VALIDATION_SIZE]

train_images = images[VALIDATION_SIZE:]
trian_labels = labels[VALIDATION_SIZE:]

#5 对训练集进行分批
batch_size = 100
n_batch = len(train_images)/batch_size

# =========================== 建立神经网络，设置损失函数，设置梯度下降的优化参数

#6,创建简单的神经网络，用来对图片进行识别
weights = tf.Variable(tf.zeros([784,10]))
biases = tf.Variable(tf.zeros([10]))
result = tf.matmul(x,weights)+biases
prediction = tf.nn.softmax(result)

# 7  创建损失函数，以交叉熵的平均值为衡量
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))

#用梯度下降法优化参数
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

'''===========3 初始化变量，设置好准确度的计算方法，在Session中运行=========='''
#9初始化变量
init = tf.global_variables_initializer()
# 10 计算准确率
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

with tf.Session() as sess:
    #初始化
    sess.run(init)
    #循环50次
    for epoch in range(0,50):
        for batch in range(0,int(n_batch)):
            #按照分片取出数据
            batch_x = train_images[batch*batch_size:(batch+1)*batch_size]
            batch_y = trian_labels[batch * batch_size:(batch + 1) * batch_size]
            #进行训练
            sess.run(train_step,feed_dict={x:batch_x,y:batch_y})
        #每一轮计算，计算一次准确度
        accuracy_n = sess.run(accuracy,feed_dict={x:validation_images,y:validation_labels})
        print("第"+str(epoch+1)+"轮，准确度为："+str(accuracy_n))











