# -*- coding: UTF-8 -*-
"""
===============================================================
author：sjyttkl
email：695492835@qq.com
date：2018
introduction:  迁移学习，论文：A Deep Convolutional Activation Featurefor Generic Visual Recognition
===============================================================
"""
__author__ = "sjyttkl"
import glob
import os.path
import random
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile

# 1. 模型和样本路径的设置
#Inception-v3模型瓶颈的节点个数
BOTTLENECK_TENSOR_SIZE = 2048
BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'
#Inception-v3模型中代表瓶颈层结果的张量名称。在谷歌提供的Inception-v3模型中，
#这个张量就是‘
JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'


MODEL_DIR = '../../datasets/inception_dec_2015'
MODEL_FILE= 'tensorflow_inception_graph.pb'

CACHE_DIR = '../../datasets/bottleneck'
INPUT_DATA = '../../datasets/flower_photos'

VALIDATION_PERCENTAGE = 10
TEST_PERCENTAGE = 10
