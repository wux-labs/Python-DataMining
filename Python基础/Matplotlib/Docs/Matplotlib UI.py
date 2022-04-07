# Databricks notebook source
# MAGIC %md
# MAGIC # Matplotlib UI

# COMMAND ----------

# MAGIC %md
# MAGIC ## 点

# COMMAND ----------

# MAGIC %md
# MAGIC ### marker
# MAGIC 
# MAGIC 绘图过程如果我们想要给坐标自定义一些不一样的标记，就可以使用 plot() 方法的 marker 参数来定义。
# MAGIC 
# MAGIC 可用的符号参数如下：
# MAGIC 
# MAGIC * 符号：'-'，'--'，'-.'，':'，'.'，','，，o，^，v，<，>，s，+，x，D，d，1，2，3，4，h，H，p，| ，_
# MAGIC * 颜色：b(蓝色)，g(绿色)，r(红色)，c(青色)，m(品红)，y(黄色)，k(黑色)，w(白色)
# MAGIC 
# MAGIC | 标记 | 描述 |
# MAGIC | ---------- | ---------- |
# MAGIC | "." | 点 |
# MAGIC | "," | 像素点 |
# MAGIC | "o" | 实心圆 |
# MAGIC | "v" | 下三角 |
# MAGIC | "^" | 上三角 |
# MAGIC | "<" | 左三角 |
# MAGIC | ">" | 右三角 |
# MAGIC | "1" | 下三叉 |
# MAGIC | "2" | 上三叉 |
# MAGIC | "3" | 左三叉 |
# MAGIC | "4" | 右三叉 |
# MAGIC | "8" | 八角形 |
# MAGIC | "s" | 正方形 |
# MAGIC | "p" | 五边形 |
# MAGIC | "P" | 加号（填充） |
# MAGIC | "*" | 星号 |
# MAGIC | "h" | 六边形 1 |
# MAGIC | "H" | 六边形 2 |
# MAGIC | "+" | 加号 |
# MAGIC | "x" | 乘号 x |
# MAGIC | "X" | 乘号 x (填充) |
# MAGIC | "D" | 菱形 |
# MAGIC | "d" | 瘦菱形 |
# MAGIC | "_" | 横线 |
# MAGIC | 0 (TICKLEFT) | 左横线 |
# MAGIC | 1 (TICKRIGHT) | 右横线 |
# MAGIC | 2 (TICKUP) | 上竖线 |
# MAGIC | 3 (TICKDOWN) | 下竖线 |
# MAGIC | 4 (CARETLEFT) | 左箭头 |
# MAGIC | 5 (CARETRIGHT) | 右箭头 |
# MAGIC | 6 (CARETUP) | 上箭头 |
# MAGIC | 7 (CARETDOWN) | 下箭头 |
# MAGIC | 8 (CARETLEFTBASE) | 左箭头 (中间点为基准) |
# MAGIC | 9 (CARETRIGHTBASE) | 右箭头 (中间点为基准) |
# MAGIC | 10 (CARETUPBASE) | 上箭头 (中间点为基准) |
# MAGIC | 11 (CARETDOWNBASE) | 下箭头 (中间点为基准) |
# MAGIC | "None", " " or "" | 没有任何标记 |
# MAGIC | '$...$' | 渲染指定的字符。例如 "$f$" 以字母 f 为标记。 |

# COMMAND ----------

from pylab import *
import numpy as np
import math

fig = plt.figure(figsize=(20,10))
x = np.arange(0, math.pi*2 + 0.1, 0.1)

plot(x, sin(x))
plot(x, cos(x), '.')
plot(x, -sin(x), marker = 'o')
plot(x, -cos(x), marker = matplotlib.markers.CARETDOWNBASE)

show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### size & color
# MAGIC 
# MAGIC 我们可以自定义标记的大小与颜色，使用的参数分别是：
# MAGIC 
# MAGIC * markersize，简写为 ms：定义标记的大小。
# MAGIC * markerfacecolor，简写为 mfc：定义标记内部的颜色。
# MAGIC * markeredgecolor，简写为 mec：定义标记边框的颜色。
# MAGIC 
# MAGIC | 颜色标记 | 描述 |
# MAGIC | ---------- | ---------- |
# MAGIC | 'r' | 红色 |
# MAGIC | 'g' | 绿色 |
# MAGIC | 'b' | 蓝色 |
# MAGIC | 'c' | 青色 |
# MAGIC | 'm' | 品红 |
# MAGIC | 'y' | 黄色 |
# MAGIC | 'k' | 黑色 |
# MAGIC | 'w' | 白色 |

# COMMAND ----------

from pylab import *
import numpy as np
import math

fig = plt.figure(figsize=(20,10))
x = np.arange(0, math.pi*2 + 0.1, 0.1)

plot(x, sin(x), marker = 'o', ms = 10)
plot(x, cos(x), marker = '.', ms = 10, mec = '#4CAF50')
plot(x, -sin(x), marker = 'o', ms = 15, mec = 'r')
plot(x, -cos(x), marker = matplotlib.markers.CARETDOWNBASE, ms = 20, mfc = 'y')

show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### fmt
# MAGIC fmt 参数定义了基本格式，如标记、线条样式和颜色。
# MAGIC ```
# MAGIC fmt = '[marker][line][color]'
# MAGIC ```
# MAGIC 例如 `o:r`，o 表示实心圆标记，: 表示虚线，r 表示颜色为红色。

# COMMAND ----------

from pylab import *
import numpy as np
import math

fig = plt.figure(figsize=(20,10))
x = np.arange(0, math.pi*2 + 0.1, 0.1)

plot(x, sin(x), 'o:b', ms = 10)
plot(x, cos(x), '.--r', ms = 10, mec = '#4CAF50')
plot(x, -sin(x), 'o', ms = 15, mec = 'r')
plot(x, -cos(x), marker = matplotlib.markers.CARETDOWNBASE, ms = 20, mfc = 'y')

show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 线
# MAGIC 绘图过程我们可以自定义线的样式，包括线的类型、颜色和大小等。
# MAGIC 
# MAGIC | 类型 | 简写 | 说明 |
# MAGIC | ---------- | ---------- | ---------- |
# MAGIC | 'solid' (默认) | '-' | 实线 |
# MAGIC | 'dotted' | ':' | 点虚线 |
# MAGIC | 'dashed' | '--' | 破折线 |
# MAGIC | 'dashdot' | '-.' | 点划线 |
# MAGIC | 'None' | '' 或 ' ' | 不画线 |

# COMMAND ----------

# MAGIC %md
# MAGIC ### 线的类型
# MAGIC 线的类型可以使用 linestyle 参数来定义，简写为 ls。

# COMMAND ----------

from pylab import *
import numpy as np
import math

fig = plt.figure(figsize=(20,10))
x = np.arange(0, math.pi*2 + 0.1, 0.1)

plot(x, sin(x), linestyle = 'dotted')
plot(x, cos(x), ls = '-.')
plot(x, -sin(x), ls = '--')
plot(x, -cos(x), ls = '-')

show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 线的颜色
# MAGIC 线的颜色可以使用 color 参数来定义，简写为 c。

# COMMAND ----------

from pylab import *
import numpy as np
import math

fig = plt.figure(figsize=(20,10))
x = np.arange(0, math.pi*2 + 0.1, 0.1)

plot(x, sin(x), linestyle = 'dotted', color = 'r')
plot(x, cos(x), ls = '-.', color = '#8FBC8F')
plot(x, -sin(x), ls = '--')
plot(x, -cos(x), ls = '-')

show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 线的宽度
# MAGIC 线的宽度可以使用 linewidth 参数来定义，简写为 lw，值可以是浮点数。

# COMMAND ----------

from pylab import *
import numpy as np
import math

fig = plt.figure(figsize=(20,10))
x = np.arange(0, math.pi*2 + 0.1, 0.1)

plot(x, sin(x), linestyle = 'dotted', color = 'r', linewidth = '5.5')
plot(x, cos(x), ls = '-.', color = '#8FBC8F', lw = '10')
plot(x, -sin(x), ls = '--')
plot(x, -cos(x), ls = '-')

show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 轴

# COMMAND ----------

# MAGIC %md
# MAGIC ### 刻度
# MAGIC Matplotlib 通过 xticks 或 yticks 来实现对坐标轴的格式设置。

# COMMAND ----------

import matplotlib.pyplot as plt
import random

plt.figure(figsize=(20,10))

# 准备数据
x = range(60)
y1 = [random.uniform(15,18) for i in x]
y2 = [random.uniform(5,8) for i in x]

# 绘制折线图
# 绘制第一条折线
plt.plot(x, y1)
# 绘制第二条折线
plt.plot(x, y2)

# 修改x、y的刻度
# 准备x的刻度说明
x_label = ["11:{}".format(i) for i in x]

# 设置x的刻度范围(0-60)及步长(5)
plt.xticks(x[::5], x_label[::5])
# 设置y的刻度范围(0-30)及步长(5)
plt.yticks(range(0, 30, 5))

plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 范围
# MAGIC 可以使用 xlim 和 ylim 来设置坐标轴的范围。

# COMMAND ----------

import matplotlib.pyplot as plt
import numpy as np

fig, axes =  plt.subplots(2,2, figsize=(15,15))

x = range(60)
y = [random.uniform(15,18) for i in x]
x_label = ["11:{}".format(i) for i in x]

axes[0][0].plot(x,y)

axes[0][1].plot(x,y)
axes[0][1].set_ylim(0,100)

axes[1][0].plot(x,y)
axes[1][0].set_xlim(0,100)

axes[1][1].plot(x,y)
axes[1][1].set_ylim(0,100)
axes[1][1].set_xlim(0,100)

plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 标签
# MAGIC 我们可以使用 xlabel() 和 ylabel() 方法来设置 x 轴和 y 轴的标签。

# COMMAND ----------

import matplotlib.pyplot as plt
import random

plt.figure(figsize=(20,10))

# 准备数据
x = range(60)
y1 = [random.uniform(15,18) for i in x]
y2 = [random.uniform(5,8) for i in x]

# 绘制折线图
# 绘制第一条折线
plt.plot(x, y1)
# 绘制第二条折线
plt.plot(x, y2)

# 修改x、y的刻度
# 准备x的刻度说明
x_label = ["11:{}".format(i) for i in x]

# 设置x的刻度范围(0-60)及步长(5)
plt.xticks(x[::5], x_label[::5])
# 设置y的刻度范围(0-30)及步长(5)
plt.yticks(range(0, 30, 5))

# 添加坐标轴标签
plt.xlabel("Time")
plt.ylabel("Temperature")

plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 标题
# MAGIC 我们可以使用 title() 方法来设置标题。

# COMMAND ----------

import matplotlib.pyplot as plt
import random

plt.figure(figsize=(20,10))

# 准备数据
x = range(60)
y1 = [random.uniform(15,18) for i in x]
y2 = [random.uniform(5,8) for i in x]

# 绘制折线图
# 绘制第一条折线
plt.plot(x, y1)
# 绘制第二条折线
plt.plot(x, y2)

# 修改x、y的刻度
# 准备x的刻度说明
x_label = ["11:{}".format(i) for i in x]

# 设置x的刻度范围(0-60)及步长(5)
plt.xticks(x[::5], x_label[::5])
# 设置y的刻度范围(0-30)及步长(5)
plt.yticks(range(0, 30, 5))

# 添加坐标轴标签
plt.xlabel("Time")
plt.ylabel("Temperature")

# 添加坐标轴标题
plt.title("Temperature Plot")

plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC 可以改变颜色、位置等。

# COMMAND ----------

import matplotlib.pyplot as plt
import random

plt.figure(figsize=(20,10))

# 准备数据
x = range(60)
y1 = [random.uniform(15,18) for i in x]
y2 = [random.uniform(5,8) for i in x]

# 绘制折线图
# 绘制第一条折线
plt.plot(x, y1)
# 绘制第二条折线
plt.plot(x, y2)

# 修改x、y的刻度
# 准备x的刻度说明
x_label = ["11:{}".format(i) for i in x]

# 设置x的刻度范围(0-60)及步长(5)
plt.xticks(x[::5], x_label[::5])
# 设置y的刻度范围(0-30)及步长(5)
plt.yticks(range(0, 30, 5))

# 添加坐标轴标签
plt.xlabel("Time", loc="right")
plt.ylabel("Temperature", loc="top")

font1 = {'color':'blue','size':20}

# 添加坐标轴标题
plt.title("Temperature Plot", fontdict = font1, loc="left")

plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 网格
# MAGIC 
# MAGIC Matplotlib 通过 grid() 方法可以开启或者关闭画布中的网格（即是否显示网格）以及网格的主/次刻度。除此之外，grid() 函数还可以设置网格的颜色、线型以及线宽等属性。

# COMMAND ----------

# MAGIC %md
# MAGIC ### pyplot.grid()
# MAGIC 
# MAGIC 我们可以使用 pyplot 中的 grid() 方法来设置图表中的网格线。
# MAGIC 
# MAGIC grid() 方法语法格式如下：
# MAGIC ```
# MAGIC matplotlib.pyplot.grid(b=None, which='major', axis='both', **kwargs)
# MAGIC ```
# MAGIC 参数说明：
# MAGIC 
# MAGIC * b：可选，默认为 None，可以设置布尔值，true 为显示网格线，false 为不显示，如果设置 **kwargs 参数，则值为 true。
# MAGIC * which：可选，可选值有 'major'、'minor' 和 'both'，默认为 'major'，表示应用更改的网格线。
# MAGIC * axis：可选，设置显示哪个方向的网格线，可以是取 'both'（默认），'x' 或 'y'，分别表示两个方向，x 轴方向或 y 轴方向。
# MAGIC * **kwargs：可选，设置网格样式，可以是 color='r', linestyle='-' 和 linewidth=2，分别表示网格线的颜色，样式和宽度。

# COMMAND ----------

# MAGIC %md
# MAGIC 以下代码给画布添加了默认的网格。

# COMMAND ----------

import matplotlib.pyplot as plt
import random

plt.figure(figsize=(20,10))

# 准备数据
x = range(60)
y1 = [random.uniform(15,18) for i in x]
y2 = [random.uniform(5,8) for i in x]

# 绘制折线图
# 绘制第一条折线
plt.plot(x, y1)
# 绘制第二条折线
plt.plot(x, y2)

# 修改x、y的刻度
# 准备x的刻度说明
x_label = ["11:{}".format(i) for i in x]

# 设置x的刻度范围(0-60)及步长(5)
plt.xticks(x[::5], x_label[::5])
# 设置y的刻度范围(0-30)及步长(5)
plt.yticks(range(0, 30, 5))

# 添加坐标轴标签
plt.xlabel("Time", loc="right")
plt.ylabel("Temperature", loc="top")

font1 = {'color':'blue','size':20}

# 添加坐标轴标题
plt.title("Temperature Plot", fontdict = font1, loc="left")

# 添加默认网格
plt.grid()

plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC 以下代码设置了网格的样式。

# COMMAND ----------

import matplotlib.pyplot as plt
import random

plt.figure(figsize=(20,10))

# 准备数据
x = range(60)
y1 = [random.uniform(15,18) for i in x]
y2 = [random.uniform(5,8) for i in x]

# 绘制折线图
# 绘制第一条折线
plt.plot(x, y1)
# 绘制第二条折线
plt.plot(x, y2)

# 修改x、y的刻度
# 准备x的刻度说明
x_label = ["11:{}".format(i) for i in x]

# 设置x的刻度范围(0-60)及步长(5)
plt.xticks(x[::5], x_label[::5])
# 设置y的刻度范围(0-30)及步长(5)
plt.yticks(range(0, 30, 5))

# 添加坐标轴标签
plt.xlabel("Time", loc="right")
plt.ylabel("Temperature", loc="top")

font1 = {'color':'blue','size':20}

# 添加坐标轴标题
plt.title("Temperature Plot", fontdict = font1, loc="left")

# 添加网格，设置只在x轴添加
plt.grid(axis='x')

plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC 以下实例添加一个简单的网格线，并设置网格线的样式，格式如下：
# MAGIC ```
# MAGIC grid(color = 'color', linestyle = 'linestyle', linewidth = number)
# MAGIC ```
# MAGIC 参数说明：
# MAGIC 
# MAGIC * color：'b' 蓝色，'m' 洋红色，'g' 绿色，'y' 黄色，'r' 红色，'k' 黑色，'w' 白色，'c' 青绿色，'#008000' RGB 颜色符串。
# MAGIC * linestyle：'‐' 实线，'‐‐' 破折线，'‐.' 点划线，':' 虚线。
# MAGIC * linewidth：设置线的宽度，可以设置一个数字。

# COMMAND ----------

import matplotlib.pyplot as plt
import random

plt.figure(figsize=(20,10))

# 准备数据
x = range(60)
y1 = [random.uniform(15,18) for i in x]
y2 = [random.uniform(5,8) for i in x]

# 绘制折线图
# 绘制第一条折线
plt.plot(x, y1)
# 绘制第二条折线
plt.plot(x, y2)

# 修改x、y的刻度
# 准备x的刻度说明
x_label = ["11:{}".format(i) for i in x]

# 设置x的刻度范围(0-60)及步长(5)
plt.xticks(x[::5], x_label[::5])
# 设置y的刻度范围(0-30)及步长(5)
plt.yticks(range(0, 30, 5))

# 添加坐标轴标签
plt.xlabel("Time", loc="right")
plt.ylabel("Temperature", loc="top")

font1 = {'color':'blue','size':20}

# 添加坐标轴标题
plt.title("Temperature Plot", fontdict = font1, loc="left")

# 添加网格，设置只在x轴添加
plt.grid(color = 'r', linestyle = '--', linewidth = 2)

plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### axes.grid()
# MAGIC axes.grid() 提供了同样的功能。
# MAGIC 
# MAGIC grid() 函数使用格式如下：
# MAGIC ```
# MAGIC grid(color='b', ls = '-.', lw = 0.25)
# MAGIC ```
# MAGIC 参数含义如下：
# MAGIC 
# MAGIC * color：表示网格线的颜色；
# MAGIC * ls：表示网格线的样式；
# MAGIC * lw：表示网格线的宽度；

# COMMAND ----------

import matplotlib.pyplot as plt
import numpy as np

fig, axes =  plt.subplots(2,2, figsize=(15,15))

x = np.arange(1,5)

#绘制平方函数
axes[0][0].plot(x,x*x)
axes[0][0].set_title('square')
axes[0][0].grid(True)

#绘制平方根图像
axes[0][1].plot(x,np.sqrt(x))
axes[0][1].set_title('square root')
axes[0][1].grid(color='b', ls = '-.', lw = 0.5)

#绘制指数函数
axes[1][0].plot(x,np.exp(x))
axes[1][0].set_title('exp')
axes[1][0].grid(color='r', ls = ':', lw = 2, axis = 'x')

#绘制对数函数
axes[1][1].plot(x,np.log10(x))
axes[1][1].set_title('log')

plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 图例
# MAGIC 在plot()函数中设置label属性后，可以使用legend()函数设置显示图例。
# MAGIC 
# MAGIC legend()函数使用格式如下: 
# MAGIC ```
# MAGIC matplotlib.pyplot.legend(*args, **kwargs)
# MAGIC ```
# MAGIC 参数含义如下：
# MAGIC 
# MAGIC | 属性 | 描述 |
# MAGIC | ---------- | ---------- |
# MAGIC | loc | Location code string, or tuple (see below).图例所有figure位置 |
# MAGIC | prop | the font property字体参数 |
# MAGIC | fontsize | the font size (used only if prop is not specified) |
# MAGIC | markerscale | the relative size of legend markers vs. original.图例标记与原始标记的相对大小 |
# MAGIC | markerfirst | If True (default), marker is to left of the label.如果为True，则图例标记位于图例标签的左侧 |
# MAGIC | numpoints | the number of points in the legend for line.为线条图图例条目创建的标记点数 |
# MAGIC | scatterpoints | the number of points in the legend for scatter plot.为散点图图例条目创建的标记点数 |
# MAGIC | scatteryoffsets | a list of yoffsets for scatter symbols in legend.为散点图图例条目创建的标记的垂直偏移量 |
# MAGIC | frameon | If True, draw the legend on a patch (frame).控制是否应在图例周围绘制框架 |
# MAGIC | fancybox | If True, draw the frame with a round fancybox.控制是否应在构成图例背景的FancyBboxPatch周围启用圆边 |
# MAGIC | shadow | If True, draw a shadow behind legend.控制是否在图例后面画一个阴影 |
# MAGIC | framealpha | Transparency of the frame.控制图例框架的 Alpha 透明度 |
# MAGIC | edgecolor | Frame edgecolor. |
# MAGIC | facecolor | Frame facecolor. |
# MAGIC | ncol | number of columns.设置图例分为n列展示 |
# MAGIC | borderpad | the fractional whitespace inside the legend border.图例边框的内边距 |
# MAGIC | labelspacing | the vertical space between the legend entries.图例条目之间的垂直间距 |

# COMMAND ----------

# MAGIC %md
# MAGIC ### 默认图例

# COMMAND ----------

import matplotlib.pyplot as plt
import numpy as np
import math

fig = plt.figure(figsize=(20,10))
x = np.arange(0, math.pi*2 + 0.1, 0.1)

plt.plot(x, sin(x),label="Sine")
plt.plot(x, cos(x), '.',label="Cosine")
plt.plot(x, -sin(x), marker = 'o',label="Inverse Sine")
plt.plot(x, -cos(x), marker = matplotlib.markers.CARETDOWNBASE,label="Inverse Cosine")

plt.legend()

plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 调整位置
# MAGIC 可以通过loc参数设置图例的位置。
# MAGIC 
# MAGIC * 0: 'best'
# MAGIC * 1: 'upper right'
# MAGIC * 2: 'upper left'
# MAGIC * 3: 'lower left'
# MAGIC * 4: 'lower right'
# MAGIC * 5: 'right'
# MAGIC * 6: 'center left'
# MAGIC * 7: 'center right'
# MAGIC * 8: 'lower center'
# MAGIC * 9: 'upper center'
# MAGIC * 10: 'center'

# COMMAND ----------

import matplotlib.pyplot as plt
import numpy as np
import math

fig = plt.figure(figsize=(20,10))
x = np.arange(0, math.pi*2 + 0.1, 0.1)

plt.plot(x, sin(x),label="Sine")
plt.plot(x, cos(x), '.',label="Cosine")
plt.plot(x, -sin(x), marker = 'o',label="Inverse Sine")
plt.plot(x, -cos(x), marker = matplotlib.markers.CARETDOWNBASE,label="Inverse Cosine")

plt.legend(loc="upper left")

plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 其他样式

# COMMAND ----------

import matplotlib.pyplot as plt
import numpy as np
import math

fig = plt.figure(figsize=(20,10))
x = np.arange(0, math.pi*2 + 0.1, 0.1)

p1 = plt.plot(x, sin(x))
p2 = plt.plot(x, cos(x), '.')
p3 = plt.plot(x, -sin(x), marker = 'o',label="Inverse Sine")
p4 = plt.plot(x, -cos(x), marker = matplotlib.markers.CARETDOWNBASE,label="Inverse Cosine")

l1 = plt.legend([p1, p2], loc="upper left")
l2 = plt.legend([p1, p2, p3, p4], loc="upper right")

plt.gca().add_artist(l1)

plt.show()
