# -*- coding: UTF-8 -*-
"""
===============================================================
author：sjyttkl
email：695492835@qq.com
date：2018
introduction: 这里是我自己加上去的。这里写的更详细
===============================================================
"""
__author__ = "sjyttkl"
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('../../datasets/MNIST_data',one_hot=True)
#每个批次的大小
batch_size = 100
#计算一共有多少个批次
n_batch = mnist.train.num_examples // batch_size

#参数概要
def variable_summariers(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean',mean)#平均值
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev',stddev)
        tf.summary.scalar('max',tf.reduce_max(var))#最大值
        tf.summary.scalar('min',tf.reduce_min(var))#最小值
        tf.summary.histogram('histogram',var)#直方图
#初始化权值
def weight_variable(shape,name):
    inital = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(inital,name=name)
#初始化偏置
def bias_variable(shape,name):
    inital = tf.constant(0.1,shape=shape)
    return tf.Variable(inital,name=name)
#卷积层
def conv2d(x,W):
    # x input tensor of shape `[batch, in_height, in_width, in_channels]`
    # W filter / kernel tensor of shape [filter_height, filter_width, in_channels, out_channels]
    # `strides[0] = strides[3] = 1`. strides[1]代表x方向的步长，strides[2]代表y方向的步长
    # padding: A `string` from: `"SAME", "VALID"`
    return tf.nn.conv2d(input=x,filter=W,strides=[1,1,1,1],padding="SAME")
#池化层
def max_pool_2x2(x):
    # 参数是四个，和卷积很类似：
    # 第一个参数value：需要池化的输入，一般池化层接在卷积层后面，所以输入通常是feature map，依然是[batch, height, width, channels]这样的shape
    # 第二个参数ksize：池化窗口的大小，取一个四维向量，一般是[1, height, width, 1]，因为我们不想在batch和channels上做池化，所以这两个维度设为了1
    # 第三个参数strides：和卷积类似，窗口在每一个维度上滑动的步长，一般也是[1, stride,stride, 1]
    # 第四个参数padding：和卷积类似，可以取'VALID' 或者'SAME'
    # 返回一个Tensor，类型不变，shape仍然是[batch, height, width, channels]这种形式
    # ksize=[1,x,y,1]
    return tf.nn.max_pool(value=x,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")

#命名空间
with tf.name_scope("input"):
    x = tf.placeholder(dtype=tf.float32,shape=[None,784],name="x-input")
    y = tf.placeholder(dtype=tf.float32,shape=[None,10],name="y-input")
    with tf.name_scope("x_image"):
        # 改变x的格式转为4D的向量[batch, in_height, in_width, in_channels]`
        x_image = tf.reshape(tensor=x,shape=[-1,28,28,1],name="x-image")

with tf.name_scope('Conv1'):
    #初始化第一个卷积层的权重和偏值
    with tf.name_scope('W_conv1'):
        W_conv1 = weight_variable([5,5,1,32],name='W_conv1')#5*5的采样窗口，32个卷积核从1个平面抽取特征
    with tf.name_scope('b_conv1'):
        b_conv1 = bias_variable([32],name='b_conv1') #每一个卷积核一个偏置值

    # 把x_image和权值向量进行卷积，再加上偏置值，然后应用于relu激活函数
    with tf.name_scope("conv2d_1"):
        conv2d_1 = conv2d(x=x_image,W=W_conv1) + b_conv1
    with tf.name_scope("relu"):
        h_conv1 = tf.nn.relu(features=conv2d_1)
    with tf.name_scope("h_pool1"):
        h_pool1 = max_pool_2x2(x=h_conv1)
with tf.name_scope("conv2d_2"):
    # 初始化第二个卷积层的权值和偏值
    with tf.name_scope(name="W_conv2"):
        W_conv2 = weight_variable(shape=[5,5,32,64],name="W_conv2")
    with tf.name_scope(name="b_conv2"):
        b_conv2 = bias_variable(shape=[64],name="b_conv2")

    # 把h_pool1 和权值进行卷积，再加上偏值，然后应用于relu激活函数
    with tf.name_scope("conv2d_2"):
        conv2d_2 = conv2d(x=x,W=W_conv2) + b_conv2
    with tf.name_scope(name="relu"):
        h_conv2 = tf.nn.relu(features=conv2d_2)
    with tf.name_scope(name="h_pool2"):
        h_pool2 = max_pool_2x2(x=h_conv2)
#28*28的图片第一次卷积后还是28*28，第一次池化后变为14*14
#第二次卷积后为14*14，第二次池化后变为了7*7
#进过上面操作后得到64张7*7的平面
with tf.name_scope("fc1"):
    #初始化第一个全连接层的权值
    with tf.name_scope("W_fc1"):
        W_fc1 = weight_variable(shape = [7*7*64,1024],name='W_fc1')#上一场有7*7*64个神经元，全连接层有1024个神经元
    with tf.name_scope("b_fc1"):
        b_fc1 = bias_variable(shape = [1024],name='b_fc1')#1024个节点

    #把池化层 2 的输出扁平化为 1 维
    with tf.name_scope("h_pool2_flat"):
        h_pool2_flat = tf.reshape(tensor=h_pool2,shape=[-1,7*7*64],name='h_pool2_flat')
    #求出第一个全连接层的输出
    with tf.name_scope("wx_plus_b1"):
        wx_plus_b1 = tf.matmul(h_pool2_flat,W_fc1) + b_fc1
    with tf.name_scope("relu"):
        h_fc1 = tf.nn.relu(wx_plus_b1)
    #keep_prob 用来表示神经网络的输出概率
    with tf.name_scope("keep_prob"):
        keep_prob = tf.placeholder(tf.float32,name="keep_prob")
    with tf.name_scope("h_fc1_drop"):
        h_fc1_drop = tf.nn.dropout(x=h_fc1,keep_prob=keep_prob,name="h_fc1_drop")

with tf.name_scope("fc2"):
    #初始化第二个全连接层
    with tf.name_scope("W_fc2"):
        W_fc2 = weight_variable(shape=[2014,10],name="W_fc2")
    with tf.name_scope("b_fc2"):
        b_fc2 = weight_variable([10],name="b_fc2")
    with tf.name_scope("wx_plux_b2"):
        wx_plus_b2 = tf.matmul(h_fc1_drop,W_fc2) + b_fc2
    with tf.name_scope("softmax"):
        #计算输出
        prediction = tf.nn.softmax(wx_plus_b2)
#交叉熵
with tf.name_scope("cross_entropy"):
    cross_entropy = tf.reduce_mean(input_tensor=tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction),name="cross_entropy")
    tf.summary.scalar(name='cross_entropy',tensor=cross_entropy)
#使用AdamOPtimizer进行优化
with tf.name_scope("train"):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
#求准确度
with tf.name_scope("accuracy"):
    with tf.name_scope("correct_prediction"):
        #结果存放在一个布尔列表中
        correct_prediction = tf.equal(tf.argmax(input=prediction,axis=1),y=tf.argmax(input=y,axis=1))#argmax返回一维张量中最大的值所在的位置
    with tf.name_scope("accuracy"):
        #求准确率
        accuracy = tf.reduce_mean(tf.cast(x=correct_prediction,dtype=tf.float32))
        tf.summary.scalar(name="accuracy",tensor=accuracy)
#合并所有的summary
merged = tf.summary.merge_all()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    train_writer = tf.summary.FileWriter(logdir='log/train',graph=sess.graph)
    test_writer = tf.summary.FileWriter(logdir='log/test',graph=sess.graph)
    for i in range(1001):
        #训练模型：
        batch_xs,batch_ys = mnist.train.next_batch(batch_size=batch_size)
        sess.run(fetches=train_step,feed_dict={x:batch_xs,y:batch_ys,keep_prob:0.7})#这里只是训练而已。
        #记录训练集里的参数
        summary = sess.run(fetches=merged,feed_dict={x:batch_xs,y:batch_ys,keep_prob:1.0})
        train_writer.add_summary(summary=summary,global_step=i)
        #记录测试集的参数
        batch_xs,batch_ys = mnist.test.next_batch(batch_size=batch_size)
        summary = sess.run(fetches=merged,feed_dict={x:batch_xs,y:batch_ys})
        test_writer.add_summary(summary=summary,global_step=i)
        if i % 100 ==0:
            test_acc = sess.run(fetches=accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels,keep_prob:1.0})
            train_acc = sess.run(fetches=accuracy,feed_dict={x:mnist.train.images,y:mnist.train.labels,keep_prob:1.0})
            print("Iter " + str(i) + ", Testing Accuracy= " + str(test_acc) + ", Training Accuracy= " + str(train_acc))













