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
import os
from tensorflow.examples.tutorials.mnist import input_data

import mnist_inference

BATCH_SIZE = 100
LEARING_RATE_BASE = 0.8
LEARING_RATE_DECAY = 0.99
RECULARAZTION_RATE = 0.0001
TRAING_STEPS = 3000
MOVING_AVERAGE_DECAY=0.99
#模型保存地址
MODEL_SAVE_PATH="../MNIST_mode/"
MODEL_NAME="mnist_model"

# 2. 定义训练过程。
def trian(mnist):
    x = tf.placeholder(dtype=tf.float32,shape=[None,mnist_inference.INPUT_NODE],name="x-input")
    y_ = tf.placeholder(dtype=tf.float32,shape=[None,mnist_inference.OUTPUT_NODE],name="y-input")

    regularizer = tf.contrib.layers.l2_regularizer(RECULARAZTION_RATE)
    y = mnist_inference.inference(x,regularizer)
    global_step = tf.Variable(0,trainable=False)
    #损失函数，学习率，滑动平均操作
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)
    variable_averages_op = variable_averages.apply(tf.trainable_variables())

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=tf.argmax(y_,1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    #指数衰减
    learning_rate = tf.train.exponential_decay(learning_rate=LEARING_RATE_BASE,
                                               global_step=global_step,
                                               decay_steps=mnist.train.num_examples/BATCH_SIZE,
                                               decay_rate=LEARING_RATE_DECAY)
    train_step = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss,global_step=global_step)
    with tf.control_dependencies([train_step,variable_averages_op]):
        train_op = tf.no_op(name="train")
    #初始化tensorflow 持久化类
    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for  i in range(TRAING_STEPS):
            xs,ys = mnist.train.next_batch(BATCH_SIZE)
            _,loss_value ,step = sess.run([train_op,loss,global_step],feed_dict={x:xs,y_:ys})
            #每100轮保存一模型
            if i % 1000 == 0:
                print("After %d trianing step(s),loss on training  batch is  %g"%(step,loss_value))
                # saver.save(sess,save_path=MODEL_SAVE_PATH,latest_filename=MODEL_NAME,global_step=global_step)
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)
def main(argv= None):
    mnist = input_data.read_data_sets("../../datasets/MNIST_data",one_hot=True)
    trian(mnist=mnist)

if __name__ =="__main__":
    tf.app.run()


