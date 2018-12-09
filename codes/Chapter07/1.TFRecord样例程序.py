# -*- coding: UTF-8 -*-
"""
===============================================================
author：sjyttkl
email：sjyttkl@jd.com
date：2018
introduction: TFRecord 文件中的数据可以通过tf.train.Example Protocol Buffer 的格式存储
===============================================================
"""
__author__ = "sjyttkl"
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
# 1. 将输入转化成TFRecord格式并保存。
# 定义函数转化变量类型。
#生成整型的人属性
def _int64_feature(value):
    return tf.train.Feature(int64_list = tf.train.Int64List(value=[value]))
#生成字符串型的属性
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
# 读取mnist数据。
# mnist=input_data.read_data_sets(train_dir="../../datasets/MNIST_data",dtype=tf.uint8,one_hot=True)
# images= mnist.train.images
# labels = mnist.train.labels
# pixels = images.shape[1]
# num_examples = mnist.train.num_examples
# #输出TFRecord文件地址
# filename= "../../datasets/path/to/output.tfrecords"
# #创建一个writer来写TFRcord文件
# writer = tf.python_io.TFRecordWriter(filename)
# for index in range(num_examples):
#     #将图像矩阵转化成一个字符串
#     image_raw = images[index].tostring()
#     # print(image_raw)
#     #将一个样例转化为Example Protocol Buffer,并将所有信息写入这个数据结构
#     example = tf.train.Example(features= tf.train.Features(feature={
#         'pixels':_int64_feature(pixels),
#         'label':_int64_feature(np.argmax(labels[index])),
#         'image_raw':_bytes_feature(image_raw)
#     }))
#     #将一个Example写入TFRecord文件
#     writer.write(example.SerializeToString())
# writer.close()


# 2. 读取TFRecord文件
reader = tf.TFRecordReader()
#创建一个队列来维护输入文件列表
file_name_queue = tf.train.string_input_producer(string_tensor=["../../datasets/path/to/output.tfrecords"])
#从文件中读取样例，也可以使用 read_up_to 函数一次性读取多个样例
_,serialized_example = reader.read(file_name_queue)
features = tf.parse_single_example(
    serialized=serialized_example,
    features={
        #tensorflow提供了两种不同的属性解析方式。一种方法是tf.FixedLenFeature,这种方法解析的结构为一个Tensor。另一种方法是：tfVarlenFeature，这种
        #方法得到的解析结果为SparseTensor，用于处理稀疏数据。这里解析数据的格式需要和上面程序写入数据格式一致
        'image_raw':tf.FixedLenFeature([],tf.string),
        'pixels':tf.FixedLenFeature([],tf.int64),
        'label':tf.FixedLenFeature([],tf.int64)
    }
)
#tf.decode_raw 可以将字符串解析成图像对应的像素数组。
images = tf.decode_raw(bytes=features['image_raw'],out_type=tf.uint8)
labels = tf.cast(features['label'],tf.int32)
pixles = tf.cast(features['pixels'],tf.int32)

sess = tf.Session()

#启动多线程处理输入数据，
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess,coord=coord)

#每次运行可以读取TFRcord文件中的一个样例，当所有样例都读取完之后，在此样例中程序会在重头读取
for i in range(10):
    image,label,pixel = sess.run([images,labels,pixles])
    print(label,pixel,images)











