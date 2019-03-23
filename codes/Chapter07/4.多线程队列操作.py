# -*- coding: utf-8 -*-

"""
==================================================
   File Name：     4.多线程队列操作
   email:         songdongdong@jd.com
   Author :       songdongdong
   date：          2019/3/23
   Description :  
==================================================
"""
__author__ = 'songdongdong'

import tensorflow as tf
#1.定义队列以及操作
queue = tf.FIFOQueue(100,"float")
enqueue_op = queue.enqueue([tf.random_normal([1])])
qr = tf.train.QueueRunner(queue,[enqueue_op]*5)
tf.train.add_queue_runner(qr)
out_tensor = queue.dequeue()

#2. 启动线程。
with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess,coord=coord)
    for _ in range(3):
        print(sess.run(out_tensor)[0])
    coord.request_stop()
    coord.join(threads)