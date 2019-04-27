# -*- coding: utf-8 -*-

"""
-------------------------------------------------
   File Name：     eax
   email:         songdongdong@jd.com
   Author :       songdongdong
   date：          2018/12/9
   Description :  
-------------------------------------------------
   Change Activity:
                   2018/12/9:
-------------------------------------------------
"""
__author__ = 'songdongdong'

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import  numpy as np
def _int64_featrue(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

mnist = input_data.read_data_sets(train_dir="../../datasets/MNIST_data",dtype=tf.uint8,one_hot=True)

images=mnist.train.images
lables = mnist.train.labels

pixels = images.shape[1]
num_examples = mnist.train.num_examples
filename= "../../datasets/path/to/output.tfrecords2"
writer = tf.python_io.TFRecordWriter(filename)
for index in range(num_examples):
    image_raw = images[index].tostring()
    print(image_raw)
    example = tf.train.Example(features= tf.train.Features(feature={
       'pixels':_int64_featrue(pixels),
        'label':_int64_featrue(np.argmax(lables[index])),
        'image_raw':_bytes_feature(image_raw)
    }))
    writer.write(example.SerializeToString())
writer.close()


reader = tf.TFRecordReader()

file_name_queue = tf.train.string_input_producer(string_tensor=["../../datasets/path/to/output.tfrecords"])

_,serialized_example = reader.read(file_name_queue)
features = tf.parse_single_example()