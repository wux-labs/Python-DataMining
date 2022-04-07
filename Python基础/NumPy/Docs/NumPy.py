# Databricks notebook source
# MAGIC %md
# MAGIC # NumPy介绍
# MAGIC NumPy（Numerical Python）是Python中科学计算的基础包。它是一个Python库，提供多维数组对象，各种派生对象（如掩码数组和矩阵），以及用于数组快速操作的各种API，有包括数学、逻辑、形状操作、排序、选择、输入输出、离散傅立叶变换、基本线性代数，基本统计运算和随机模拟等等。
# MAGIC 
# MAGIC NumPy包的核心是 ndarray 对象。它封装了Python原生的同数据类型的 n 维数组，为了保证其性能优良，其中有许多操作都是代码在本地进行编译后执行的。
# MAGIC 
# MAGIC 简而言之：
# MAGIC * NumPy是一个开源的Python科学计算库，用于快速处理任意维度的数组。  
# MAGIC * NumPy支持常见的数组和矩阵操作。对于同样的数值计算任务，使用NumPy比直接使用Python要简洁的多。  
# MAGIC * NumPy使用ndarray对象来处理多维数组，该对象是一个快速而灵活的大数据容器。

# COMMAND ----------

# MAGIC %md
# MAGIC ## ndarray介绍
# MAGIC NumPy提供了一个**N维数组类型ndarray**，它描述了相同类型的items的集合。

# COMMAND ----------

import numpy as np

score = np.array([[80,89,86,87],[78,79,89,67],[90,94,78,67]])

print(type(score))

print(score)

# COMMAND ----------

# MAGIC %md
# MAGIC ## ndarray与Python原生list运算效率对比
# MAGIC 在这里我们通过一段代码运行来体会到ndarray的好处。

# COMMAND ----------

import random
import time
import numpy as np

a = []
for i in range(100000000):
    a.append(random.random())

# COMMAND ----------

# 使用Python原生list进行运算
t1 = time.time()
sum1 = sum(a)
t2 = time.time()

# 使用ndarray进行运算
b = np.array(a)
t3 = time.time()
sum2 = np.sum(b)
t4 = time.time()

# 考察两种方式的处理时间
print(t2 - t1, t4 - t3)

# COMMAND ----------

# MAGIC %md
# MAGIC 从中我们看到ndarray的计算速度快很多，节约了时间。  
# MAGIC **机器学习的最大特点就是大量的数据运算**，那么如果没有一个快速的解决方案，那可能现在Python也在机器学习领域达不到号的效果。  
# MAGIC NumPy专门针对ndarray的操作和运算进行了设计，所以数组的存储效率和输入输出性能远优于Python中的嵌套列表，数组越大，NumPy的优势就越明显。

# COMMAND ----------

# MAGIC %md
# MAGIC ## ndarray的优势
# MAGIC * 内存块风格，ndarray存储的是相同类型的数据，所以在内存中是连续存储的。  
# MAGIC * ndarray支持并行化运算。  
# MAGIC * NumPy底层使用C语言编写，内部解除了GIL（全局解释器锁），其对数组的操作速度不受Python解释器的限制，效率远高于Python代码。

# COMMAND ----------

# MAGIC %md
# MAGIC # 认识N维数组

# COMMAND ----------

# MAGIC %md
# MAGIC ## ndarray的属性
# MAGIC 数组属性反映了数组本身固有的信息。  
# MAGIC * `ndarray.shape`：数组维度的元组
# MAGIC * `ndarray.ndim`：数组维数
# MAGIC * `ndarray.size`：数组中元素的数量
# MAGIC * `ndarray.itemsize`：一个数组元素的长度（字节）
# MAGIC * `ndarray.dtype`：数组元素的类型

# COMMAND ----------

import numpy as np

score = np.array([[80,89,86,87],[78,79,89,67],[90,94,78,67]])

print(score.shape, score.ndim, score.size, score.dtype, score.itemsize)

# COMMAND ----------

# MAGIC %md
# MAGIC # 基本操作
# MAGIC * 生成数组的方法
# MAGIC * 数组的索引、切片
# MAGIC * 形状修改
# MAGIC * 类型修改
# MAGIC * 数组去重

# COMMAND ----------

# MAGIC %md
# MAGIC ## 生成数组的方法

# COMMAND ----------

# MAGIC %md
# MAGIC ### 生成有初始占位符内容的数组
# MAGIC * `np.zeros(shape)` 生成0
# MAGIC * `np.ones(shape)` 生成1
# MAGIC * `np.empty(shape)` 生成随机数

# COMMAND ----------

import numpy as np

np.zeros(shape=(3,4), dtype="int32")

# COMMAND ----------

import numpy as np

np.ones(shape=(2,3), dtype=np.int32)

# COMMAND ----------

import numpy as np

np.empty(shape=(2,3), dtype=np.float64)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 从现有数组中生成
# MAGIC * `np.array()` 深拷贝
# MAGIC * `np.copy()` 深拷贝
# MAGIC * `np.asarray()` 浅拷贝

# COMMAND ----------

import numpy as np

score = np.array([[80,89,86,87],[78,79,89,67],[90,94,78,67]])

score

# COMMAND ----------

data1 = np.array(score)

data1

# COMMAND ----------

data2 = np.copy(score)

data2

# COMMAND ----------

data3 = np.asarray(score)

data3

# COMMAND ----------

# MAGIC %md
# MAGIC 改变score中的元素后，注意观察 data1、data2、data3 的数据情况，体会深拷贝、浅拷贝。

# COMMAND ----------

score[1, 2] = 1000

# COMMAND ----------

print("data1:",data1)
print("data2:",data2)
print("data3:",data3)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 生成固定范围的数组

# COMMAND ----------

# MAGIC %md
# MAGIC #### numpy.arange()
# MAGIC numpy.arange() 函数创建数值范围并返回 ndarray 对象，函数格式如下：
# MAGIC ```
# MAGIC numpy.arange(start, stop, step, dtype)
# MAGIC ```
# MAGIC 
# MAGIC 根据 start 与 stop 指定的范围以及 step 设定的步长，生成一个 ndarray。
# MAGIC 
# MAGIC 参数说明：
# MAGIC * start 	起始值，默认为0
# MAGIC * stop 	终止值（不包含）
# MAGIC * step 	步长，默认为1
# MAGIC * dtype 	返回ndarray的数据类型，如果没有提供，则会使用输入数据的类型。

# COMMAND ----------

import numpy as np
print(np.arange(10))
print(np.arange(2,10))
print(np.arange(0,100,11))
print(np.arange(0,100,11,float))

# COMMAND ----------

# MAGIC %md
# MAGIC #### numpy.linspace()
# MAGIC numpy.linspace()函数用于创建一个一维数组，数组是一个等差数列构成的，格式如下：
# MAGIC ```
# MAGIC np.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None)
# MAGIC ```
# MAGIC 
# MAGIC 生成等间隔的序列。
# MAGIC 
# MAGIC * start 序列的起始值
# MAGIC * stop 序列的终止值，如果endpoint为true，该值包含于序列中
# MAGIC * num 要生成的等间隔样例数，默认为50
# MAGIC * endpoint 序列中是否包含stop值，默认为true
# MAGIC * retstep 如果为true，返回样例以及连续数字之间的步长
# MAGIC * dtype 输出ndarray的数据类型

# COMMAND ----------

import numpy as np
# 返回2到100的数组
print(np.linspace(2,100))
# 返回0到100、步长为11的数组
print(np.linspace(0,100,11))
# 不包含最后一个数字
print(np.linspace(0,100,11, False))
# 返回样例及步长
print(np.linspace(0,100,11, True, True))
# 设置数据类型
print(np.linspace(0,100,11, True, True, int))


# COMMAND ----------

# MAGIC %md
# MAGIC #### numpy.logspace()
# MAGIC numpy.logspace()函数用于创建一个于等比数列。格式如下：
# MAGIC ```
# MAGIC np.logspace(start, stop, num=50, endpoint=True, base=10.0, dtype=None)
# MAGIC ```
# MAGIC base 参数意思是取对数的时候 log 的下标。
# MAGIC 
# MAGIC * start 	序列的起始值为：base ** start
# MAGIC * stop 	序列的终止值为：base ** stop。如果endpoint为true，该值包含于数列中
# MAGIC * num 	要生成的等步长的样本数量，默认为50
# MAGIC * endpoint 	该值为 true 时，数列中中包含stop值，反之不包含，默认是True。
# MAGIC * base 	对数 log 的底数。
# MAGIC * dtype 	ndarray 的数据类型

# COMMAND ----------

import numpy as np

# 默认底数是10
print(np.logspace(1.0,  2.0, num = 10))
print(np.logspace(1.0,  10.0, base = 2))


# COMMAND ----------

# MAGIC %md
# MAGIC ### 生成随机数组
# MAGIC * np.random
# MAGIC   * 均匀分布（每组的可能性相等）
# MAGIC     * np.random.rand(d0, d1, d2, ..., dn)
# MAGIC     * np.random.uniform(low=0.0, high=1.0, size=None)
# MAGIC     * np.random.randint(low, high=None, size=None, dtype="l")
# MAGIC   * 正太分布
# MAGIC     * np.random.normal(loc=0.0, scale=1.0, size=None)
# MAGIC     * np.random.randn(d0, d1, d2, ..., dn)
# MAGIC     * no.random.standard_normal(size=None)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 均匀分布

# COMMAND ----------

# MAGIC %md
# MAGIC ##### numpy.random.rand()
# MAGIC 创建一个给定类型的数组，将其填充在一个均匀分布的随机样本[0, 1)中。
# MAGIC 
# MAGIC ```
# MAGIC numpy.random.rand(d0, d1, d2, ..., dn)
# MAGIC ```

# COMMAND ----------

import numpy as np
import matplotlib.pyplot as plt

# 生成一个1维数组，第1维的元素个数是2
print(np.random.rand(2))
# 生成一个2维数组，第1维的元素个数是2，第2维的元素个数是3
print(np.random.rand(2, 3))
# 生成一个3维数组，第1维的元素个数是2，第2维的元素个数是3，第3维的元素个数是4
print(np.random.rand(2, 3, 4))
# 生成一个3维数组，第1维的元素个数是3，第2维的元素个数是3，第3维的元素个数是4
print(np.random.rand(3, 3, 4))

plt.hist(np.random.rand(3000))
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ##### numpy.random.uniform()
# MAGIC 
# MAGIC 从一个均匀分布[low,high)中随机采样，注意定义域是左闭右开，即包含low，不包含high。
# MAGIC 
# MAGIC ```
# MAGIC np.random.uniform(low=0.0, high=1.0, size=None)
# MAGIC ```
# MAGIC 
# MAGIC 参数介绍: 
# MAGIC     
# MAGIC * low: 采样下界，float类型，默认值为0；
# MAGIC * high: 采样上界，float类型，默认值为1；
# MAGIC * size: 输出样本数目，为int或元组(tuple)类型，例如，size=(m,n,k), 则输出m*n*k个样本，缺省时输出1个值。

# COMMAND ----------

import numpy as np
import matplotlib.pyplot as plt

# 在 [0,2) 中随机采样1个
print(np.random.uniform(2))
# 在 [2,10) 中随机采样1个
print(np.random.uniform(2, 10))
# 在 [2,10) 中随机采样8个
print(np.random.uniform(2, 10, 8))

# 在 [1,40) 中随机采样500个
plt.hist(np.random.uniform(low=1, high=40, size=500))
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### 正太分布

# COMMAND ----------

# MAGIC %md
# MAGIC ##### numpy.random.normal()
# MAGIC 从正态（高斯）分布中抽取随机样本。
# MAGIC ```
# MAGIC numpy.random.normal(loc=0.0, scale=1.0, size=None)
# MAGIC ```
# MAGIC 
# MAGIC * loc 分布的均值
# MAGIC * scale 分布的标准差
# MAGIC * size 取样个数

# COMMAND ----------

import numpy as np
import matplotlib.pyplot as plt

print(np.random.normal(1, 0.5, 1000))

plt.hist(np.random.normal(10, 2, 1000))
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ##### numpy.random.randn()
# MAGIC randn函数返回一个或一组样本，具有标准正态分布。
# MAGIC ```
# MAGIC numpy.random.randn(d0,d1,…,dn)
# MAGIC ```
# MAGIC 
# MAGIC * dn表格每个维度
# MAGIC * 返回值为指定维度的array

# COMMAND ----------

import numpy as np
import matplotlib.pyplot as plt

# 生成一个1维数组，第1维的元素个数是2
print(np.random.randn(2))
# 生成一个2维数组，第1维的元素个数是2，第2维的元素个数是3
print(np.random.randn(2, 3))
# 生成一个3维数组，第1维的元素个数是2，第2维的元素个数是3，第3维的元素个数是4
print(np.random.randn(2, 3, 4))
# 生成一个3维数组，第1维的元素个数是3，第2维的元素个数是3，第3维的元素个数是4
print(np.random.randn(3, 3, 4))

plt.hist(np.random.randn(3000))
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 数组运算
# MAGIC 数组上的算术运算符会应用到 元素 级别。

# COMMAND ----------

from pylab import *
import numpy as np
import math
import matplotlib.pyplot as plt

x = np.arange(0, math.pi*2 + 0.1, 0.1)

print("原始数组为：", x)
print("求正弦：", sin(x))
print("求余弦：", cos(x))
print("正弦 加 原数组：", sin(x) + x)
print("3 加 原数组：", 3 + x)
print("求和：", x.sum())
print("求最大值：", x.max())

plt.plot(x, label="origin")
plt.plot(sin(x), label="sine")
plt.plot(cos(x), label="cosine")
plt.plot(sin(x) + x, label="add sine")
plt.plot(3 + x, label="add 3")
plt.legend()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 逻辑运算

# COMMAND ----------

# MAGIC %md
# MAGIC ### 统计运算

# COMMAND ----------

# MAGIC %md
# MAGIC ### 数组间运算

# COMMAND ----------

# MAGIC %md
# MAGIC #### 数组与数的运算

# COMMAND ----------

# MAGIC %md
# MAGIC #### 数组与数组的运算

# COMMAND ----------

# MAGIC %md
# MAGIC #### 矩阵的运算

# COMMAND ----------

# MAGIC %md
# MAGIC ## 通函数
# MAGIC NumPy提供熟悉的数学函数，例如sin，cos和exp。在NumPy中，这些被称为“通函数”（ufunc）。在NumPy中，这些函数在数组上按元素进行运算，产生一个数组作为输出。

# COMMAND ----------

import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 6.4, 0.1)

print("原始数组为：", x)
print("求正弦：", np.sin(x))
print("求余弦：", np.cos(x))
print("求指数：", np.exp(x) + x)
print("3 加 原数组：", 3 + x)
print("求和：", x.sum())
print("求最大值：", x.max())

plt.plot(x, label="origin")
plt.plot(np.sin(x), label="sine")
plt.plot(np.cos(x), label="cosine")
plt.plot(np.exp(x) + x, label="exp")
plt.plot(3 + x, label="add 3")
plt.legend()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 数组的索引、切片、迭代
# MAGIC ndarray对象的内容可以通过索引或切片来访问和修改，与 Python 中 list 的切片操作一样。
# MAGIC 
# MAGIC ndarray 数组可以基于 0 - n 的下标进行索引，切片对象可以通过内置的 slice 函数，并设置 start, stop 及 step 参数进行，从原数组中切割出一个新数组。

# COMMAND ----------

# MAGIC %md
# MAGIC ### 索引

# COMMAND ----------

import numpy as np
import matplotlib.pyplot as plt

a1 = np.random.rand(20)
a2 = np.random.rand(4, 5)
a3 = np.random.rand(6, 7, 8)

print("原始数组", a1)
# 按下标进行索引
print("下标10的元素", a1[10])

print("原始数组", a2)
# 按下标进行索引
print("下标1的元素", a2[1])
# 按下标进行索引
print("下标1,3的元素", a2[1][3])

print("原始数组", a3)
# 按下标进行索引
print("下标1,3,5的元素", a3[1][3][5])


# COMMAND ----------

# MAGIC %md
# MAGIC ### 切片

# COMMAND ----------

import numpy as np
import matplotlib.pyplot as plt

a1 = np.random.rand(20)
a2 = np.random.rand(4, 5)
a3 = np.random.rand(6, 7, 8)

print("原始数组", a1)
print("数组切片", a1[1:3])
print("数组切片", a1[5:])

print("原始数组", a3)
print("数组切片", a3[1:])
print("数组切片", a3[1:2])


# COMMAND ----------

# MAGIC %md
# MAGIC ### 迭代

# COMMAND ----------

import numpy as np
import matplotlib.pyplot as plt

a1 = np.random.rand(20)

for a in a1:
    print(a)

print("------------------------------")

a2 = np.random.rand(4, 5)

for a in a2:
    print(a)

print("------------------------------")

for a in a2.flat:
    print(a)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 形状操作

# COMMAND ----------

# MAGIC %md
# MAGIC ### 改变数组的形状
# MAGIC 一个数组的形状是由每个轴的元素数量决定的

# COMMAND ----------

# MAGIC %md
# MAGIC #### reshape()
# MAGIC 不会改变原始数组，而是返回一个新的数组

# COMMAND ----------

import numpy as np
import matplotlib.pyplot as plt

a1 = np.random.rand(20)

print(a1)
print(a1.shape)

a2 = a1.reshape((4,5))
print(a2)
print(a2.shape)


# COMMAND ----------

# MAGIC %md
# MAGIC #### resize()
# MAGIC 不会返回新的数组，而是在原始数组的基础上进行修改。

# COMMAND ----------

import numpy as np
import matplotlib.pyplot as plt

a1 = np.random.rand(20)

print(a1)
print(a1.shape)

a1.resize((4,5))

print(a1)
print(a1.shape)

# COMMAND ----------

# MAGIC %md
# MAGIC #### T
# MAGIC 可以进行数组的行列转置。不会修改原始数组，而是返回一个新的数组。

# COMMAND ----------

import numpy as np
import matplotlib.pyplot as plt

a1 = np.random.rand(4, 5)

print(a1)
print(a1.shape)

a2 = a1.T

print(a2)
print(a2.shape)

print(a1)
print(a1.shape)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 将不同数组堆叠在一起
# MAGIC 几个数组可以沿不同的轴堆叠在一起。

# COMMAND ----------

import numpy as np
import matplotlib.pyplot as plt

a1 = np.random.rand(3, 4)
a2 = np.random.rand(3, 4)

print(a1)
print(a2)

print(np.vstack((a1, a2)))
print(np.hstack((a1, a2)))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 将一个数组拆分成几个较小的数组
# MAGIC 
# MAGIC 使用hsplit ，可以沿数组的水平轴拆分数组，方法是指定要返回的形状相等的数组的数量，或者指定应该在哪些列之后进行分割。
# MAGIC 
# MAGIC vsplit 沿垂直轴分割，并array_split 允许指定要分割的轴。

# COMMAND ----------

import numpy as np
import matplotlib.pyplot as plt

a1 = np.random.rand(6, 12)

print(a1)

# 拆分成两个大小相同的数组
a2,a3 = np.hsplit(a1,2)

print(a2)
print(a3)

# 根据第2、5、10三个列的位置，将数组拆分成4个小数组
a4,a5,a6,a7 = np.hsplit(a1,(2,5,10))
print(a4)
print(a5)
print(a6)
print(a7)


# COMMAND ----------

# MAGIC %md
# MAGIC # IO操作与数据处理
