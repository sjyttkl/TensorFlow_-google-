# -*- coding: UTF-8 -*-
"""
===============================================================
author：sjyttkl
email：695492835@qq.com
date：2018
introduction:
===============================================================
"""

# 一、numpy.flatten
import numpy as np
a = np.array([[1,2], [3,4]])
print(a.flatten())   # 默认参数为"C"，即按照行进行重组
print(a.flatten('F')) # 按照列进行重组