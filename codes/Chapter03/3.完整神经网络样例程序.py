# -*- coding: UTF-8 -*-
"""
===============================================================
author：songdongdong
email：songdongdong@jd.com
date：2018
introduction:
===============================================================
"""
import tensorflow as tf
from numpy.random  import RandomState

#1. 定义神经网络的参数，输入和输出节点。
batch_size = 8
w1 = tf.Variable(tf.random_normal([2,3],stddev=1,seed=1))
w2 = tf.Variable(tf.random_normal([3,1],stddev=1,seed=1))
x = tf.placeholder(tf.float32,shape=[None,2],name="x-input")
y_ = tf.placeholder(tf.float32,shape=[None,1],name="y-input")

# 2. 定义前向传播过程，损失函数及反向传播算法。
a = tf.matmul(x,w1)
y = tf.matmul(a,w2)
cross_entry = -tf.reduce_mean(y_*tf.log(tf.clip_by_value(y,1e-10,1.0))) #clip_by_value是为了限制最大，最小范围
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entry)

# 3. 生成模拟数据集。
rdm = RandomState(1)
X= rdm.rand(128,2)#(128,2)
Y=[[int(x1+x2 <1)] for (x1,x2) in X] #(128,1)

#4. 创建一个会话来运行TensorFlow程序。


with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    # 输出目前（未经训练）的参数取值。
    print("w1:", sess.run(w1))
    print("w2:", sess.run(w2))
    print("\n")

    #训练模型
    STEPS = 500
    for i in range(0,STEPS):
        start = (i*batch_size)%128
        end = (i*batch_size)%128+batch_size
        sess.run(train_step,feed_dict={x: X[start:end], y_: Y[start:end]})
        if 1 %1000 == 0:
            total_cross_entropy = sess.run(cross_entry,feed_dict={x:X,y_:y})
            print("After %d training step(s), cross entropy on all data is %g" % (i, total_cross_entropy))

        # 输出训练后的参数取值。
    print("\n")
    print("w1:", sess.run(w1))
    print("w2:", sess.run(w2))



