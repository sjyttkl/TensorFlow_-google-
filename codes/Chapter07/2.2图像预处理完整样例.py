# -*- coding: utf-8 -*-

"""
==================================================
   File Name：     2.2图像预处理完整样例
   email:         songdongdong@jd.com
   Author :       songdongdong
   date：          2019/3/23
   Description :  
==================================================
"""
__author__ = 'songdongdong'

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
# 1. 随机调整图片的色彩，定义两种顺序。
def distort_color(image,color_ordering=0):
    if color_ordering == 0 :
        image = tf.image.random_brightness(image=image,max_delta=32./255.) #随机亮度调整在[-max_delta,max_delta]范围内
        image = tf.image.random_saturation(image=image,lower=0.5,upper=1.5)#饱和度
        image = tf.image.random_hue(image=image,max_delta=0.2)#色相
        image = tf.image.random_contrast(image=image,lower=0.5,upper=1.5)  #对比度
    else:
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)  #饱和度
        image = tf.image.random_brightness(image, max_delta=32. / 255.)#随机亮度调整在[-max_delta,max_delta]范围内
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5) #对比度
        image = tf.image.random_hue(image, max_delta=0.2) #色相

    return tf.clip_by_value(image, 0.0, 1.0)  #将张量值剪切到指定的最小值和最大值

#2. 对图片进行预处理，将图片转化成神经网络的输入层数据。
def preprocess_for_train(image, height, width, bbox):
    # 查看是否存在标注框。#如果没有提供标注框，则认为整个图像就是需要关注的部分
    if bbox is None:
        bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
    # 转换图像张量的类型
    if image.dtype != tf.float32:
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    # 随机的截取图片中一个块。#随机截取图像，减少需要关注的物体大小对图像识别的影响
    bbox_begin, bbox_size, _ = tf.image.sample_distorted_bounding_box(
        tf.shape(image), bounding_boxes=bbox)
    bbox_begin, bbox_size, _ = tf.image.sample_distorted_bounding_box(
        tf.shape(image), bounding_boxes=bbox)


    distorted_image = tf.slice(image, bbox_begin, bbox_size)

    # 将随机截取的图片调整为神经网络输入层的大小。# 大小调整的算法是随机的
    distorted_image = tf.image.resize_images(distorted_image, [height, width], method=np.random.randint(4))
    distorted_image = tf.image.random_flip_left_right(distorted_image)
    # 使用一种随机的顺序调整图像色彩
    distorted_image = distort_color(distorted_image, np.random.randint(2))
    return distorted_image
#3. 读取图片。
image_raw_data = tf.gfile.FastGFile("../../datasets/flower_photos/daisy/172967318_c596d082cc.jpg","rb").read()
with tf.Session() as sess:
    img_data = tf.image.decode_jpeg(image_raw_data)
    boxes = tf.constant([[[0.05, 0.05, 0.9, 0.7], [0.35, 0.47, 0.5, 0.56]]])
    # 运行6次获得6中不同的图像，在图中显示效果
    for i in range(9):
        # 将图像的尺寸调整为299*299
        result = preprocess_for_train(img_data, 299, 299, boxes)
        plt.imshow(result.eval())
        plt.show()