# -*- coding: UTF-8 -*-
"""
===============================================================
author：sjyttkl
email：695492835@qq.com
date：2018
introduction:
===============================================================
"""
'''
二、TensorFlow的基础运算

在搞神经网络之前，先让我们把TensorFlow的基本运算，也就是加减乘除搞清楚。

首先，TensorFlow有几个概念需要进行明确：

1 图（Graph）：用来表示计算任务，也就我们要做的一些操作。

2 会话（Session）：建立会话，此时会生成一张空图；在会话中添加节点和边，形成一张图，一个会话可以有多个图，通过执行这些图得到结果。如果把每个图看做一个车床，那会话就是一个车间，里面有若干个车床，用来把数据生产成结果。

3 Tensor：用来表示数据，是我们的原料。

4 变量（Variable）：用来记录一些数据和状态，是我们的容器。

5 feed和fetch：可以为任意的操作(arbitrary operation) 赋值或者从其中获取数据。相当于一些铲子，可以操作数据。

形象的比喻是：把会话看做车间，图看做车床，里面用Tensor做原料，变量做容器，feed和fetch做铲子，把数据加工成我们的结果。


链接：https://www.jianshu.com/p/2ea7a0632239


'''
# 2.1 创建图和运行图：
import tensorflow as tf
v1 = tf.constant([[2,3]])
v2 = tf.constant(([2],[3]))
product = tf.matmul(v1,v2)
print(product)
sess = tf.Session()
result = sess.run(product)
print(result)
sess.close()
#================================================
# 2.2 创建一个变量，并用for循环对变量进行赋值操作
#创建一个变量num
num = tf.Variable(0,name="count")
#创建一个加法操作
new_value = tf.add(num,10)
#这是一个赋值操作，把new_value赋值给new_value
op = tf.assign(num,new_value)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(num))
    for i in range(0,5):
        sess.run(op)
        print(sess.run(num))

# 2.3 通过feed设置placeholder的值

input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
new_value = tf.multiply(input1,input2)

with tf.Session() as sess:
    print(sess.run(new_value,feed_dict={input1:23.0,input2:11.0}))


#
import numpy as np
# 首先，明确一点，tf.argmax可以认为就是np.argmax。tensorflow使用numpy实现的这个API。
# 　　
# 简单的说，tf.argmax就是返回最大的那个数值所在的下标。
# 　　
# 这个很好理解，只是tf.argmax()的参数让人有些迷惑，比如，tf.argmax(array, 1)和tf.argmax(array, 0)有啥区别呢？
test = np.array([[1, 2, 3], [2, 3, 4], [5, 4, 3], [8, 7, 2]])
np.argmax(test, 0)      #输出 ：array([3, 3, 1])
np.argmax(test, 1)     #输出：array([2, 2, 0, 0])
'''
axis = 0: 
　　你就这么想，0是最大的范围，所有的数组都要进行比较，只是比较的是这些数组相同位置上的数：
    test[0] = array([1, 2, 3])
    test[1] = array([2, 3, 4])
    test[2] = array([5, 4, 3])
    test[3] = array([8, 7, 2])
    # output   :    [3, 3, 1]  
axis = 1: 
　　等于1的时候，比较范围缩小了，只会比较每个数组内的数的大小，结果也会根据有几个数组，产生几个结果。
    test[0] = array([1, 2, 3])  #2
    test[1] = array([2, 3, 4])  #2
    test[2] = array([5, 4, 3])  #0
    test[3] = array([8, 7, 2])  #0
注意：
   这是里面都是数组长度一致的情况，如果不一致，axis最大值为最小的数组长度-1，超过则报错。 
　　当不一致的时候，axis=0的比较也就变成了每个数组的和的比较。

实际上是这样的:
把数组按下图的矩阵模式排列，axis=0就是求矩阵x轴方向所有y轴最大值下标，axis=1是求矩阵y轴方向所有x轴最大值的下标。
(下面[表情][表情]不显示，所以用l代替)
# l y方向          axis=1的时候，返回沿着y轴的所有x最大值下标。    
# l [1, 2, 3]    -->2                   
# l [2, 3, 4]    -->2    
# l [5, 4, 3]    -->0    
# l [8, 7 ,2]    -->0    
# --------------------------> x方向    
#  [3, 3, 1]  <-- 当axis=0的时候，返回沿着x轴的所有y值的最大下标。 

'''
# tf.equal的使用
print("tf.equal的使用")
import tensorflow as tf
import numpy as np
#tf.equal(A, B)是对比这两个矩阵或者向量的相等的元素，如果是相等的那就返回True，反正返回False，返回的值的矩阵维度和A是一样的
A = [[1, 3, 4, 5, 6],[2,3,2,1,3]]
B = [[1, 3, 4, 3, 2]]

with tf.Session() as sess:
    print(sess.run(tf.equal(A, B)))

#tf.cast的使用
print("tf.cast的使用")
# cast(x, dtype, name=None)
# 将x的数据格式转化成dtype.例如，原来x的数据格式是bool，
# 那么将其转化成float以后，就能够将其转化成0和1的序列。反之也可以
a = tf.Variable([1,0,0,1,1])
b = tf.cast(a,dtype=tf.bool)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
print(sess.run(b))
sess.close()
#[ True False False  True  True]
c = tf.cast(b,dtype=tf.float32)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
print(sess.run(c))
sess.close()
