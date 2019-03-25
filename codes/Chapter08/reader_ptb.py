# -*- coding: utf-8 -*-

"""
==================================================
   File Name：     reader_ptb
   email:         songdongdong@jd.com
   Author :       songdongdong
   date：          2019/3/24
   Description :  
==================================================
"""
__author__ = 'songdongdong'

# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


"""Utilities for parsing PTB text files."""

import collections
import os

import numpy as np
import tensorflow as tf


def _read_words(filename):
  with tf.gfile.GFile(filename, "r") as f:  #gfile.GFile主要是用于HDFS等文件系统中的文件操作
    return f.read().replace("\n", "<eos>").split() #读取文件， 将换行符替换为 <eos>， 然后将文件按空格分割。 返回一个 1-D list


def _build_vocab(filename):  #用于建立字典
  data = _read_words(filename)
  counter = collections.Counter(data) #输出一个字典： key是word， value是这个word出现的次数
  count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
#counter.items() 会返回一个tuple列表， tuple是(key, value), 按 value的降序，key的升序排列
  words, _ = list(zip(*count_pairs)) #感觉这个像unzip 就是把key放在一个tuple里，value放在一个tuple里
  word_to_id = dict(zip(words, range(len(words))))#对每个word进行编号， 按照之前words输出的顺序（value降序，key升序）
  return word_to_id  #返回dict， key：word， value：id


def _file_to_word_ids(filename, word_to_id): #将file表示为word_id的形式
  data = _read_words(filename)
  return [word_to_id[word] for word in data]

def ptb_raw_data(data_path=None):
  """Load PTB raw data from data directory "data_path".
  Reads PTB text files, converts strings to integer ids,
  and performs mini-batching of the inputs.
  The PTB dataset comes from Tomas Mikolov's webpage:
  http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
  Args:
    data_path: string path to the directory where simple-examples.tgz has
      been extracted.
  Returns:
    tuple (train_data, valid_data, test_data, vocabulary)
    where each of the data objects can be passed to PTBIterator.
  """

  train_path = os.path.join(data_path, "ptb.train.txt")
  valid_path = os.path.join(data_path, "ptb.valid.txt")
  test_path = os.path.join(data_path, "ptb.test.txt")

  word_to_id = _build_vocab(train_path) #使用训练集确定word id
  train_data = _file_to_word_ids(train_path, word_to_id)
  valid_data = _file_to_word_ids(valid_path, word_to_id)
  test_data = _file_to_word_ids(test_path, word_to_id)
  vocabulary = len(word_to_id)#字典的大小
  return train_data, valid_data, test_data, vocabulary


def ptb_iterator(raw_data, batch_size, num_steps):
  """Iterate on the raw PTB data.
  This generates batch_size pointers into the raw PTB data, and allows
  minibatch iteration along these pointers.
  Args:
    raw_data: one of the raw data outputs from ptb_raw_data.
    batch_size: int, the batch size.
    num_steps: int, the number of unrolls.
  Yields:
    Pairs of the batched data, each a matrix of shape [batch_size, num_steps].
    The second element of the tuple is the same data time-shifted to the
    right by one.
  Raises:
    ValueError: if batch_size or num_steps are too high.
  """
  raw_data = np.array(raw_data, dtype=np.int32)#raw data : train_data | vali_data | test data

  data_len = len(raw_data) #how many words in the data_set
  batch_len = data_len // batch_size
  data = np.zeros([batch_size, batch_len], dtype=np.int32)#batch_len 就是几个word的意思
  for i in range(batch_size):
    data[i] = raw_data[batch_len * i:batch_len * (i + 1)]

  epoch_size = (batch_len - 1) // num_steps

  if epoch_size == 0:
    raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

  for i in range(epoch_size):
    x = data[:, i*num_steps:(i+1)*num_steps]
    y = data[:, i*num_steps+1:(i+1)*num_steps+1]
  yield (x, y)


def ptb_producer(raw_data, batch_size, num_steps, name=None):
    """Iterate on the raw PTB data.
    This chunks up raw_data into batches of examples and returns Tensors that
    are drawn from these batches.
    Args:
      raw_data: one of the raw data outputs from ptb_raw_data.
      batch_size: int, the batch size.
      num_steps: int, the number of unrolls.
      name: the name of this operation (optional).
    Returns:
      A pair of Tensors, each shaped [batch_size, num_steps]. The second element
      of the tuple is the same data time-shifted to the right by one.
    Raises:
      tf.errors.InvalidArgumentError: if batch_size or num_steps are too high.
    """
    with tf.name_scope(name, "PTBProducer", [raw_data, batch_size, num_steps]):
        # 原始数据就是一个个的单词，这里将原始数据转换为tensor
        raw_data = tf.convert_to_tensor(raw_data, name="raw_data", dtype=tf.int32)
        # 求单词的个数
        data_len = tf.size(raw_data)
        # 得到总共批的个数
        batch_len = data_len // batch_size
        # 将样本进行reshape
        # shape的行数是一个批的大小，最后处理的时候是一列一列处理的
        # shape的列数是总共批的个数
        data = tf.reshape(raw_data[0: batch_size * batch_len],
                          [batch_size, batch_len])

        # epoch_size是用总的批数除以时间步长长度
        # 得到的就是运行一个epoch需要运行num_steps的个数
        epoch_size = (batch_len - 1) // num_steps
        #tf.assert_positive如果x中有负数就抛出异常，
        assertion = tf.assert_positive(epoch_size,message="epoch_size == 0, decrease batch_size or num_steps")#messageepoch_size=(batch_len - 1) // num_steps="epoch_size == 0, decrease batch_size or num_steps"
        """
        y = tf.identity(x)是一个op操作表示将x的值赋予y
        y = x只是一个内存拷贝，并不是一个op，而control_dependencies只有当里面是一个Op的时候才能起作用。
        Assign, Identity 这两个OP 与Variable 关系极其紧密，分别实现了变量的修改与读取。因此，它们必须与Variable 在同一个设备上执行；这样的关系，常称为同位关系(Colocation)。
        """
        with tf.control_dependencies([assertion]):
            epoch_size = tf.identity(epoch_size, name="epoch_size")

        # 产生一个队列，队列的长度为epoch_size，未对样本打乱
        # i是一个出列的操作，每次出列1，也就是一个num_steps
        i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
        # 将数据进行切片，起始点是[0, i * num_steps]
        # 终止点是[batch_size, (i + 1) * num_steps]
        # 其中终止点的batch_size代表的是维度
        # (i + 1) * num_steps代表的是数据的长度
        # 这里即将data数据从第i * num_steps列开始，向后取(i + 1) * num_steps列，即一个num_steps的长度
        x = tf.strided_slice(data, [0, i * num_steps],
                             [batch_size, (i + 1) * num_steps])
        # 将取到的数据reshape一下
        x.set_shape([batch_size, num_steps])
        # y的切法和x类似，只是y要向后一列移动一个单位，因为这里是根据上一个单词预测下一个单词
        y = tf.strided_slice(data, [0, i * num_steps + 1],
                             [batch_size, (i + 1) * num_steps + 1])
        y.set_shape([batch_size, num_steps])
    # 返回为两个tensor,一个是输入数据，一个是输出标签（因为任务是预测下一个单词，所以标签只是输入向后平移一个单词）
    return x,y