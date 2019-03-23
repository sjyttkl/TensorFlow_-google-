# -*- coding: utf-8 -*-

"""
==================================================
   File Name：     5.输入文件队列
   email:         songdongdong@jd.com
   Author :       songdongdong
   date：          2019/3/24
   Description :  
==================================================
"""
__author__ = 'songdongdong'

import tensorflow as tf

#1. 生成文件存储样例数据。
#创建TFRcord文件帮助函数
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

num_shards = 2#总共写入多少个文件
instances_per_shard = 2 #定义每个文件多少个数据
for i in range(num_shards):
    #将数据分为多个的时候，可以按照命名规范进行分区。前面是第几个分区，后面的是一共多少个分区
    filename = ('Records/data.tfrecords-%.5d-of-%.5d' % (i, num_shards))
    # 将Example结构写入TFRecord文件。
    writer = tf.python_io.TFRecordWriter(filename)
    for j in range(instances_per_shard):
    # Example结构仅包含当前样例属于第几个文件以及是当前文件的第几个样本。
        example = tf.train.Example(features=tf.train.Features(feature={
            'i': _int64_feature(i),
            'j': _int64_feature(j)}))
        writer.write(example.SerializeToString())
    writer.close()

# 2. 读取文件。
files = tf.train.match_filenames_once("Records/data.tfrecords-*")#获取文件列表
#通过tf.train.string_input_producer函数创建输入队，输入队列中的文件列表为tf.train.match_filenames_once函数获取的文件列表
filename_queue = tf.train.string_input_producer(files, shuffle=False)
reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)
features = tf.parse_single_example(
      serialized_example,
      features={
          'i': tf.FixedLenFeature([], tf.int64),
          'j': tf.FixedLenFeature([], tf.int64),
      })
with tf.Session() as sess:
    #虽然没有定义任何变量，但是tf.train.match_filenames_once函数时需要初始化一些变量
    tf.global_variables_initializer().run()
    print (sess.run(files))

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    for i in range(6):
        print (sess.run([features['i'], features['j']]))
    coord.request_stop()
    coord.join(threads)



#3. 组合训练数据（Batching）
example, label = features['i'], features['j']
batch_size = 2#batch中的数量，batch大写和队列大小息息相关，下面是设置队列大小的方式。太大的话，占有资源。太小，会因为没有数据进程要等待
capacity = 1000 + 3 * batch_size

#当队列长度等于容量的时候，会等待出队。当元素个数小于容量时候，Tensorflow将会自动重启动入队
example_batch, label_batch = tf.train.batch([example, label], batch_size=batch_size, capacity=capacity)

#min_after_dequeue函数提供限制出队时最少元素的个数来保证随机打乱顺序的作用。当函数被调用但是队列中元素不够时，出队操作将等待更多的元素
# 入队才能完成。
# example_batch, label_batch = tf.train.shuffle_batch([example, label], batch_size=batch_size, capacity=capacity,min_after_dequeue=30)

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    #获取打印组合后的样例，
    for i in range(3):
        cur_example_batch, cur_label_batch = sess.run([example_batch, label_batch])
        print (cur_example_batch, cur_label_batch)
    coord.request_stop()
    coord.join(threads)
