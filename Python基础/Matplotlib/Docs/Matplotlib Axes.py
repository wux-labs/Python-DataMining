# Databricks notebook source
# MAGIC %md
# MAGIC # Matplotlib Axes
# MAGIC 
# MAGIC Matplotlib 定义了一个 axes 类（轴域类），该类的对象被称为 axes 对象（即轴域对象），它指定了一个有数值范围限制的绘图区域。在一个给定的画布（figure）中可以包含多个 axes 对象，但是同一个 axes 对象只能在一个画布中使用。
# MAGIC 
# MAGIC > 2D 绘图区域（axes）包含两个轴（axis）对象；如果是 3D 绘图区域，则包含三个。

# COMMAND ----------

from matplotlib import pyplot as plt

# COMMAND ----------

# MAGIC %md
# MAGIC ## add_axes
# MAGIC 
# MAGIC 通过调用 add_axes() 方法能够将 axes 对象添加到画布中，该方法用来生成一个 axes 轴域对象，对象的位置由参数rect决定。
# MAGIC 
# MAGIC rect 是位置参数，接受一个由 4 个元素组成的浮点数列表，形如 [left, bottom, width, height] ，它表示添加到画布中的矩形区域的左下角坐标(x, y)，以及宽度和高度。

# COMMAND ----------

figure=plt.figure()
ax=figure.add_axes([0,0,1,1])

plt.show()

# COMMAND ----------

figure=plt.figure()
ax=figure.add_axes([0,0,1,1])
ax=figure.add_axes([0.5,0.5,1,1])

plt.show()

# COMMAND ----------

figure=plt.figure()
ax = figure.gca(projection='3d')

plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## subplot
# MAGIC 
# MAGIC matplotlib.pyplot模块提供了一个 subplot() 函数，它可以均等地划分画布。

# COMMAND ----------

plt.plot([1,2,3])

#现在创建一个子图，它表示一个有2行1列的网格的顶部图。
#因为这个子图将与第一个重叠，所以之前创建的图将被删除
plt.subplot(221)
plt.plot(range(12))

#创建带有黄色背景的第二个子图
plt.subplot(222, facecolor='y')
plt.plot(range(12))

#创建带有红色背景的第三个子图
plt.subplot(325, facecolor='r')
plt.plot(range(12))

# COMMAND ----------

# MAGIC %md
# MAGIC 如果不想覆盖之前的图，需要使用 add_subplot() 函数

# COMMAND ----------

fig = plt.figure()

ax1 = fig.add_subplot(111)
ax1.plot([1,2,3])

ax2 = fig.add_subplot(221, facecolor='y')
ax2.plot([1,2,3])

# COMMAND ----------

# MAGIC %md
# MAGIC 通过给画布添加 axes 对象可以实现在同一画布中插入另外的图像。

# COMMAND ----------

import matplotlib.pyplot as plt
import numpy as np
import math

x = np.arange(0, math.pi*2, 0.05)

fig=plt.figure()
y = np.sin(x)

axes1 = fig.add_axes([0.1, 0.1, 0.8, 0.8]) # main axes
axes2 = fig.add_axes([0.55, 0.55, 0.3, 0.3]) # inset axes
axes1.plot(x, y, 'b')
axes2.plot(x,np.cos(x),'r')
axes1.set_title('sine')
axes2.set_title("cosine")

plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## subplots
# MAGIC 
# MAGIC matplotlib.pyplot模块提供了一个 subplots() 函数，它的使用方法和 subplot() 函数类似。其不同之处在于，subplots() 既创建了一个包含子图区域的画布，又创建了一个 figure 图形对象，而 subplot() 只是创建一个包含子图区域的画布。
# MAGIC ```
# MAGIC figure, axes = plt.subplots(nrows=1,ncols=2,figsize=(20,8))
# MAGIC ```
# MAGIC nrows 与 ncols 表示两个整数参数，它们指定子图所占的行数、列数。
# MAGIC 
# MAGIC 函数的返回值是一个元组，包括一个图形对象和所有的 axes 对象。其中 axes 对象的数量等于 nrows * ncols，且每个 axes 对象均可通过索引值访问。

# COMMAND ----------

# MAGIC %md
# MAGIC 下面我们创建了一个 2 行 2 列的子图，并在每个子图中显示 4 个不同的图像。

# COMMAND ----------

import matplotlib.pyplot as plt
import numpy as np

fig, a =  plt.subplots(2,2)

x = np.arange(1,5)

#绘制平方函数
a[0][0].plot(x,x*x)
a[0][0].set_title('square')

#绘制平方根图像
a[0][1].plot(x,np.sqrt(x))
a[0][1].set_title('square root')

#绘制指数函数
a[1][0].plot(x,np.exp(x))
a[1][0].set_title('exp')

#绘制对数函数
a[1][1].plot(x,np.log10(x))
a[1][1].set_title('log')

plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## subplot2grid
# MAGIC matplotlib.pyplot 模块提供了 subplot2grid() ，该函数能够在画布的特定位置创建 axes 对象（即绘图区域）。不仅如此，它还可以使用不同数量的行、列来创建跨度不同的绘图区域。与 subplot() 和 subplots() 函数不同，subplot2gird() 函数以非等分的形式对画布进行切分，并按照绘图区域的大小来展示最终绘图结果。
# MAGIC 
# MAGIC 函数语法格式如下：
# MAGIC ```
# MAGIC plt.subplot2grid(shape, location, rowspan, colspan)
# MAGIC ```
# MAGIC 参数含义如下：
# MAGIC * shape：把该参数值规定的网格区域作为绘图区域；
# MAGIC * location：在给定的位置绘制图形，初始位置 (0,0) 表示第1行第1列；
# MAGIC * rowsapan/colspan：这两个参数用来设置让子区跨越几行几列。

# COMMAND ----------

import matplotlib.pyplot as plt
import numpy as np

#使用 colspan指定列，使用rowspan指定行
a1 = plt.subplot2grid((3,3),(0,0),colspan = 2)
a2 = plt.subplot2grid((3,3),(0,2), rowspan = 3)
a3 = plt.subplot2grid((3,3),(1,0),rowspan = 2, colspan = 2)

x = np.arange(1,10)

a1.plot(x, np.exp(x))
a1.set_title('exp')

a2.plot(x, x*x)
a2.set_title('square')

a3.plot(x, np.log(x))
a3.set_title('log')

plt.tight_layout()
plt.show()
