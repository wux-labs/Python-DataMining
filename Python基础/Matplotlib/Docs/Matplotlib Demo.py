# Databricks notebook source
# MAGIC %md
# MAGIC # Matplotlib Demo

# COMMAND ----------

# MAGIC %md
# MAGIC ## 折线图
# MAGIC 折线图（line chart）是我们日常工作、学习中经常使用的一种图表，它可以直观的反映数据的变化趋势。与绘制柱状图、饼状图等图形不同，Matplotlib 并没有直接提供绘制折线图的函数。

# COMMAND ----------

# 引入 matplotlib.pyplot 模块
import matplotlib.pyplot as plt

# 引入其他需要的模块
import random

# 1、创建画布（容器层）
plt.figure(figsize=(20,8))

# 2、绘制折线图（图像层）
plt.plot([1,2,3,4,5,6,7],[17,18,17,15,11,11,13])

# 3、显示图像
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 散点图（scatter）
# MAGIC 散点图是指在回归分析中，数据点在直角坐标系平面上的分布图，散点图表示因变量随自变量而变化的大致趋势，据此可以选择合适的函数对数据点进行拟合。
# MAGIC 
# MAGIC 用两组数据构成多个坐标点，考察坐标点的分布，判断两变量之间是否存在某种关联或总结坐标点的分布模式。散点图将序列显示为一组点。值由点在图表中的位置表示。类别由图表中的不同标记表示。散点图通常用于比较跨类别的聚合数据。
# MAGIC 
# MAGIC 我们可以使用 pyplot 中的 scatter() 方法来绘制散点图。
# MAGIC 
# MAGIC scatter() 方法语法格式如下：
# MAGIC ```
# MAGIC matplotlib.pyplot.scatter(x, y, s=None, c=None, marker=None, cmap=None, norm=None, vmin=None, vmax=None, alpha=None, linewidths=None, *, edgecolors=None, plotnonfinite=False, data=None, **kwargs)
# MAGIC ```
# MAGIC 
# MAGIC 参数说明：
# MAGIC 
# MAGIC * x，y：长度相同的数组，也就是我们即将绘制散点图的数据点，输入数据。
# MAGIC * s：点的大小，默认 20，也可以是个数组，数组每个参数为对应点的大小。
# MAGIC * c：点的颜色，默认蓝色 'b'，也可以是个 RGB 或 RGBA 二维行数组。
# MAGIC * marker：点的样式，默认小圆圈 'o'。
# MAGIC * cmap：Colormap，默认 None，标量或者是一个 colormap 的名字，只有 c 是一个浮点数数组的时才使用。如果没有申明就是 image.cmap。
# MAGIC * norm：Normalize，默认 None，数据亮度在 0-1 之间，只有 c 是一个浮点数的数组的时才使用。
# MAGIC * vmin，vmax：：亮度设置，在 norm 参数存在时会忽略。
# MAGIC * alpha：：透明度设置，0-1 之间，默认 None，即不透明。
# MAGIC * linewidths：：标记点的长度。
# MAGIC * edgecolors：：颜色或颜色序列，默认为 'face'，可选值有 'face', 'none', None。
# MAGIC * plotnonfinite：：布尔值，设置是否使用非限定的 c ( inf, -inf 或 nan) 绘制点。
# MAGIC * **kwargs：：其他参数。

# COMMAND ----------

# 引入 matplotlib.pyplot 模块
import matplotlib.pyplot as plt

# 引入其他需要的模块
import random

# 1、创建画布（容器层）
plt.figure(figsize=(20,8))

# 2、绘制散点图（图像层）
plt.scatter(range(50),[random.uniform(1,20) for i in range(50)])

# 3、显示图像
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 设置图标大小

# COMMAND ----------

# 引入 matplotlib.pyplot 模块
import matplotlib.pyplot as plt

# 引入其他需要的模块
import random

# 1、创建画布（容器层）
plt.figure(figsize=(20,8))

# 2、绘制散点图（图像层）
plt.scatter(range(50),[random.uniform(1,20) for i in range(50)],[random.uniform(60,500) for i in range(50)])

# 3、显示图像
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 设置图标颜色

# COMMAND ----------

# 引入 matplotlib.pyplot 模块
import matplotlib.pyplot as plt

# 引入其他需要的模块
import random

# 1、创建画布（容器层）
plt.figure(figsize=(20,8))

# 2、绘制散点图（图像层）
plt.scatter(x=range(50),y=[random.uniform(1,20) for i in range(50)],s=[random.uniform(60,500) for i in range(50)],c=[random.uniform(1,50) for i in range(50)])

# 3、显示图像
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 设置颜色条
# MAGIC 设置颜色条需要使用 cmap 参数，默认值为 'viridis'，之后颜色值设置为 0 到 100 的数组。
# MAGIC 
# MAGIC colorbar()函数用来设置显示颜色条。

# COMMAND ----------

# 引入 matplotlib.pyplot 模块
import matplotlib.pyplot as plt

# 引入其他需要的模块
import random

# 1、创建画布（容器层）
plt.figure(figsize=(20,8))

# 2、绘制散点图（图像层）
plt.scatter(x=range(50),y=[random.uniform(1,20) for i in range(50)],s=[random.uniform(60,500) for i in range(50)],c=[random.uniform(1,50) for i in range(50)])

plt.colorbar()

# 3、显示图像
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### 换个颜色
# MAGIC 设置颜色条需要使用 cmap 参数，默认值为 'viridis'，之后颜色值设置为 0 到 100 的数组。
# MAGIC 
# MAGIC 可选的颜色条：
# MAGIC 
# MAGIC | 颜色名称 | 保留关键字 |
# MAGIC | ---------- | ---------- |
# MAGIC | Accent | Accent_r |
# MAGIC | Blues | Blues_r |
# MAGIC | BrBG | BrBG_r |
# MAGIC | BuGn | BuGn_r |
# MAGIC | BuPu | BuPu_r |
# MAGIC | CMRmap | CMRmap_r |
# MAGIC | Dark2 | Dark2_r |
# MAGIC | GnBu | GnBu_r |
# MAGIC | Greens | Greens_r |
# MAGIC | Greys | Greys_r |
# MAGIC | OrRd | OrRd_r |
# MAGIC | Oranges | Oranges_r |
# MAGIC | PRGn | PRGn_r |
# MAGIC | Paired | Paired_r |
# MAGIC | Pastel1 | Pastel1_r |
# MAGIC | Pastel2 | Pastel2_r |
# MAGIC | PiYG | PiYG_r |
# MAGIC | PuBu | PuBu_r |
# MAGIC | PuBuGn | PuBuGn_r |
# MAGIC | PuOr | PuOr_r |
# MAGIC | PuRd | PuRd_r |
# MAGIC | Purples | Purples_r |
# MAGIC | RdBu | RdBu_r |
# MAGIC | RdGy | RdGy_r |
# MAGIC | RdPu | RdPu_r |
# MAGIC | RdYlBu | RdYlBu_r |
# MAGIC | RdYlGn | RdYlGn_r |
# MAGIC | Reds | Reds_r |
# MAGIC | Set1 | Set1_r |
# MAGIC | Set2 | Set2_r |
# MAGIC | Set3 | Set3_r |
# MAGIC | Spectral | Spectral_r |
# MAGIC | Wistia | Wistia_r |
# MAGIC | YlGn | YlGn_r |
# MAGIC | YlGnBu | YlGnBu_r |
# MAGIC | YlOrBr | YlOrBr_r |
# MAGIC | YlOrRd | YlOrRd_r |
# MAGIC | afmhot | afmhot_r |
# MAGIC | autumn | autumn_r |
# MAGIC | binary | binary_r |
# MAGIC | bone | bone_r |
# MAGIC | brg | brg_r |
# MAGIC | bwr | bwr_r |
# MAGIC | cividis | cividis_r |
# MAGIC | cool | cool_r |
# MAGIC | coolwarm | coolwarm_r |
# MAGIC | copper | copper_r |
# MAGIC | cubehelix | cubehelix_r |
# MAGIC | flag | flag_r |
# MAGIC | gist_earth | gist_earth_r |
# MAGIC | gist_gray | gist_gray_r |
# MAGIC | gist_heat | gist_heat_r |
# MAGIC | gist_ncar | gist_ncar_r |
# MAGIC | gist_rainbow | gist_rainbow_r |
# MAGIC | gist_stern | gist_stern_r |
# MAGIC | gist_yarg | gist_yarg_r |
# MAGIC | gnuplot | gnuplot_r |
# MAGIC | gnuplot2 | gnuplot2_r |
# MAGIC | gray | gray_r |
# MAGIC | hot | hot_r |
# MAGIC | hsv | hsv_r |
# MAGIC | inferno | inferno_r |
# MAGIC | jet | jet_r |
# MAGIC | magma | magma_r |
# MAGIC | nipy_spectral | nipy_spectral_r |
# MAGIC | ocean | ocean_r |
# MAGIC | pink | pink_r |
# MAGIC | plasma | plasma_r |
# MAGIC | prism | prism_r |
# MAGIC | rainbow | rainbow_r |
# MAGIC | seismic | seismic_r |
# MAGIC | spring | spring_r |
# MAGIC | summer | summer_r |
# MAGIC | tab10 | tab10_r |
# MAGIC | tab20 | tab20_r |
# MAGIC | tab20b | tab20b_r |
# MAGIC | tab20c | tab20c_r |
# MAGIC | terrain | terrain_r |
# MAGIC | twilight | twilight_r |
# MAGIC | twilight_shifted | twilight_shifted_r |
# MAGIC | viridis | viridis_r |
# MAGIC | winter | winter_r |

# COMMAND ----------

# 引入 matplotlib.pyplot 模块
import matplotlib.pyplot as plt

# 引入其他需要的模块
import random

# 1、创建画布（容器层）
plt.figure(figsize=(20,8))

# 2、绘制散点图（图像层）
plt.scatter(x=range(50),y=[random.uniform(1,20) for i in range(50)],s=[random.uniform(60,500) for i in range(50)],c=[random.uniform(1,50) for i in range(50)], cmap='afmhot_r')

plt.colorbar()

# 3、显示图像
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 柱状图
# MAGIC 柱状图是一种用矩形柱来表示数据分类的图表，柱状图可以垂直绘制，也可以水平绘制，它的高度与其所表示的数值成正比关系。柱状图显示了不同类别之间的比较关系，图表的水平轴 X 指定被比较的类别，垂直轴 Y 则表示具体的类别值。
# MAGIC 
# MAGIC Matplotlib 提供了bar()函数来绘制柱状图，它可以应用在 MATLAB 样式以及面向对象的绘图方法中。其语法格式如下：
# MAGIC ```
# MAGIC ax.bar(x, height, width, bottom, align)
# MAGIC ```
# MAGIC 
# MAGIC 参数说明：
# MAGIC 
# MAGIC * x：浮点型数组，柱形图的 x 轴数据。
# MAGIC * height：浮点型数组，柱形图的高度。
# MAGIC * width：浮点型数组，柱形图的宽度。
# MAGIC * bottom：浮点型数组，底座的 y 坐标，默认 0。
# MAGIC * align：柱形图与 x 坐标的对齐方式，'center' 以 x 位置为中心，这是默认值。 'edge'：将柱形图的左边缘与 x 位置对齐。要对齐右边缘的条形，可以传递负数的宽度值及 align='edge'。
# MAGIC * **kwargs：：其他参数。
# MAGIC 
# MAGIC 该函数的返回值是一个 Matplotlib 容器对象，该对象包含了所有柱状图。

# COMMAND ----------

# 引入 matplotlib.pyplot 模块
import matplotlib.pyplot as plt

# 引入其他需要的模块
import random

# 1、创建画布（容器层）
plt.figure(figsize=(20,8))

# 2、绘制柱状图（图像层）
plt.bar(range(10),[random.uniform(1,20) for i in range(10)])

# 3、显示图像
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 设置柱状方向
# MAGIC 垂直方向的柱形图可以使用 barh() 方法来设置。

# COMMAND ----------

# 引入 matplotlib.pyplot 模块
import matplotlib.pyplot as plt

# 引入其他需要的模块
import random

# 1、创建画布（容器层）
plt.figure(figsize=(20,8))

# 2、绘制柱状图（图像层）
plt.barh(range(10),[random.uniform(1,20) for i in range(10)])

# 3、显示图像
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 调整柱状宽度

# COMMAND ----------

# 引入 matplotlib.pyplot 模块
import matplotlib.pyplot as plt

# 引入其他需要的模块
import random

# 1、创建画布（容器层）
plt.figure(figsize=(20,8))

# 2、绘制柱状图（图像层）
plt.bar(range(10),[random.uniform(1,20) for i in range(10)], width = 0.2)

# 3、显示图像
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 设置多组柱状
# MAGIC 通过调整柱状图的宽度，可以实现在同一 x 轴位置绘制多个柱状图。您可以将它们设置成不同的颜色，从而使它们更容易区分。

# COMMAND ----------

# 引入 matplotlib.pyplot 模块
import matplotlib.pyplot as plt

# 引入其他需要的模块
import random
import numpy as np

# 1、创建画布（容器层）
plt.figure(figsize=(20,8))

# 2、绘制柱状图（图像层）
plt.bar(range(10),[random.uniform(1,20) for i in range(10)], width = 0.2)
plt.bar(np.arange(10)+0.2,[random.uniform(1,20) for i in range(10)], width = 0.2)
plt.bar(np.arange(10)+0.4,[random.uniform(1,20) for i in range(10)], width = 0.2)

# 3、显示图像
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 设置堆叠柱状
# MAGIC 所谓堆叠柱状图就是将不同数组别的柱状图堆叠在一起，堆叠后的柱状图高度显示了两者相加的结果值。
# MAGIC 
# MAGIC bar() 函数提供了一个可选参数bottom，该参数可以指定柱状图开始堆叠的起始值，一般从底部柱状图的最大值开始，依次类推。

# COMMAND ----------

# 引入 matplotlib.pyplot 模块
import matplotlib.pyplot as plt

# 引入其他需要的模块
import random
import numpy as np

# 1、创建画布（容器层）
plt.figure(figsize=(20,8))

y1 = np.array([random.uniform(1,20) for i in range(10)])
y2 = np.array([random.uniform(1,20) for i in range(10)])
y3 = np.array([random.uniform(1,20) for i in range(10)])

# 2、绘制柱状图（图像层）
plt.bar(range(10),y1, bottom=y2+y3)
plt.bar(range(10),y2, bottom=y3)
plt.bar(range(10),y3)

# 3、显示图像
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 直方图
# MAGIC 直方图（Histogram），又称质量分布图，它是一种条形图的一种，由一系列高度不等的纵向线段来表示数据分布的情况。 直方图的横轴表示数据类型，纵轴表示分布情况。
# MAGIC 
# MAGIC 首先，我们需要了解柱状图和直方图的区别。直方图用于概率分布，它显示了一组数值序列在给定的数值范围内出现的概率；而柱状图则用于展示各个类别的频数。
# MAGIC 
# MAGIC 如果想要构建直方图，必须遵循以下步骤：
# MAGIC 
# MAGIC * 将整个值范围划分为一系列区间。
# MAGIC * 区间值（bin）的取值，不可遗漏数据；
# MAGIC * 计算每个区间中有多少个值。
# MAGIC 
# MAGIC 通常将 bin 指定为连续且不重叠的数值区间，而 bin 值指区间开始和结束的数值。
# MAGIC 您可以使用下面的函数来绘制直方图：
# MAGIC ```
# MAGIC matplotlib.pyplot.hist()
# MAGIC ```
# MAGIC 
# MAGIC 该函数的参数说明如下：
# MAGIC 
# MAGIC * x 	必填参数，数组或者数组序列。
# MAGIC * bins 	可选参数，整数或者序列，bins 表示每一个间隔的边缘（起点和终点）默认会生成10个间隔。
# MAGIC * range 	指定全局间隔的下限与上限值 (min,max)，元组类型，默认值为 None。
# MAGIC * density 	如果为 True，返回概率密度直方图；默认为 False，返回相应区间元素的个数的直方图。
# MAGIC * histtype 	要绘制的直方图类型，默认值为“bar”，可选值有 barstacked(堆叠条形图)、step(未填充的阶梯图)、stepfilled(已填充的阶梯图)。

# COMMAND ----------

# 引入 matplotlib.pyplot 模块
import matplotlib.pyplot as plt

# 引入其他需要的模块
import random
import numpy as np

# 1、创建画布（容器层）
plt.figure(figsize=(20,8))

y1 = np.array([random.uniform(1,20) for i in range(50)])

# 2、绘制直方图（图像层）
plt.hist(y1)

# 3、显示图像
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 设置图形

# COMMAND ----------

# 引入 matplotlib.pyplot 模块
import matplotlib.pyplot as plt

# 引入其他需要的模块
import random
import numpy as np

# 1、创建画布（容器层）
plt.figure(figsize=(20,8))

y1 = np.array([random.uniform(1,20) for i in range(50)])

# 2、绘制直方图（图像层）
plt.hist(y1, histtype="step")

# 3、显示图像
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 饼状图
# MAGIC 饼状图用来显示一个数据系列，具体来说，饼状图显示一个数据系列中各项目的占项目总和的百分比。
# MAGIC 
# MAGIC Matplotlib 提供了一个 pie() 函数，该函数可以生成数组中数据的饼状图。您可使用 x/sum(x) 来计算各个扇形区域占饼图总和的百分比。
# MAGIC 
# MAGIC pie() 方法语法格式如下：
# MAGIC ```
# MAGIC matplotlib.pyplot.pie(x, explode=None, labels=None, colors=None, autopct=None, pctdistance=0.6, shadow=False, labeldistance=1.1, startangle=0, radius=1, counterclock=True, wedgeprops=None, textprops=None, center=0, 0, frame=False, rotatelabels=False, *, normalize=None, data=None)[source]
# MAGIC ```
# MAGIC 
# MAGIC 参数说明：
# MAGIC 
# MAGIC * x：浮点型数组，表示每个扇形的面积。
# MAGIC * explode：数组，表示各个扇形之间的间隔，默认值为0。
# MAGIC * labels：列表，各个扇形的标签，默认值为 None。
# MAGIC * colors：数组，表示各个扇形的颜色，默认值为 None。
# MAGIC * autopct：设置饼图内各个扇形百分比显示格式，%d%% 整数百分比，%0.1f 一位小数， %0.1f%% 一位小数百分比， %0.2f%% 两位小数百分比。
# MAGIC * labeldistance：标签标记的绘制位置，相对于半径的比例，默认值为 1.1，如 <1则绘制在饼图内侧。
# MAGIC * pctdistance：类似于 labeldistance，指定 autopct 的位置刻度，默认值为 0.6。
# MAGIC * shadow：布尔值 True 或 False，设置饼图的阴影，默认为 False，不设置阴影。
# MAGIC * radius：设置饼图的半径，默认为 1。
# MAGIC * startangle：起始绘制饼图的角度，默认为从 x 轴正方向逆时针画起，如设定 =90 则从 y 轴正方向画起。
# MAGIC * counterclock：布尔值，设置指针方向，默认为 True，即逆时针，False 为顺时针。
# MAGIC * wedgeprops：字典类型，默认值 None。参数字典传递给 wedge 对象用来画一个饼图。例如：wedgeprops={'linewidth':5} 设置 wedge 线宽为5。
# MAGIC * textprops：字典类型，默认值为：None。传递给 text 对象的字典参数，用于设置标签（labels）和比例文字的格式。
# MAGIC * center：浮点类型的列表，默认值：(0,0)。用于设置图标中心位置。
# MAGIC * frame：布尔类型，默认值：False。如果是 True，绘制带有表的轴框架。
# MAGIC * rotatelabels：布尔类型，默认为 False。如果为 True，旋转每个 label 到指定的角度。

# COMMAND ----------

# 引入 matplotlib.pyplot 模块
import matplotlib.pyplot as plt

# 引入其他需要的模块
import random
import numpy as np

# 1、创建画布（容器层）
plt.figure(figsize=(20,8))

list = [random.uniform(1,5) for i in range(10)]

# 2、绘制饼状图（图像层）
plt.pie(list)

# 3、显示图像
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 设置标签

# COMMAND ----------

# 引入 matplotlib.pyplot 模块
import matplotlib.pyplot as plt

# 引入其他需要的模块
import random
import numpy as np

# 1、创建画布（容器层）
plt.figure(figsize=(20,8))

list = [random.uniform(1,5) for i in range(10)]

# 2、绘制饼状图（图像层）
plt.pie(list,labels=["label {}".format(i) for i in range(10)])

# 3、显示图像
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 设置颜色

# COMMAND ----------

# 引入 matplotlib.pyplot 模块
import matplotlib.pyplot as plt

# 引入其他需要的模块
import random
import numpy as np

# 1、创建画布（容器层）
plt.figure(figsize=(20,8))

list = [random.uniform(1,5) for i in range(10)]

# 2、绘制饼状图（图像层）
plt.pie(list,labels=["label {}".format(i) for i in range(10)],colors=["#{}{}{}".format(1*i, 5*i, 3*i) for i in range(10,20)])

# 3、显示图像
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 突出展示

# COMMAND ----------

# 引入 matplotlib.pyplot 模块
import matplotlib.pyplot as plt

# 引入其他需要的模块
import random
import numpy as np

# 1、创建画布（容器层）
plt.figure(figsize=(20,8))

list = [random.uniform(1,5) for i in range(10)]

# 2、绘制饼状图（图像层）
plt.pie(list,labels=["label {}".format(i) for i in range(10)], explode=[((random.uniform(0,0.5) + 0.7) // 1) * random.uniform(0,0.5) for i in range(10)])

# 3、显示图像
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 等高线
# MAGIC 等高线图（也称“水平图”）是一种在二维平面上显示 3D 图像的方法。等高线有时也被称为 “Z 切片”，如果您想要查看因变量 Z 与自变量 X、Y 之间的函数图像变化（即 Z=f(X,Y)），那么采用等高线图最为直观。
# MAGIC 
# MAGIC 自变量 X 和 Y 需要被限制在矩形网格内，您可以将 x 、y 数组作为参数传递给 numpy.meshgrid() 函数来构建一个网格点矩阵。

# COMMAND ----------

import numpy as np
import matplotlib.pyplot as plt
#创建xlist、ylist数组
xlist = np.linspace(-3.0, 3.0, 100)
ylist = np.linspace(-3.0, 3.0, 100)
#将上述数据变成网格数据形式
X, Y = np.meshgrid(xlist, ylist)
#定义Z与X,Y之间的关系
Z = np.sqrt(X**2 + Y**2)
fig,ax=plt.subplots(1,1)
#填充等高线颜色
cp = ax.contourf(X, Y, Z)
fig.colorbar(cp) # 给图像添加颜色柱
ax.set_title('Filled Contours Plot')
ax.set_xlabel('x (cm)')
ax.set_ylabel('y (cm)')
#画等高线
plt.contour(X,Y,Z)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 箱型图
# MAGIC 箱型图（也称为盒须图）于 1977 年由美国著名统计学家约翰·图基（John Tukey）发明。它能显示出一组数据的最大值、最小值、中位数、及上下四分位数。
# MAGIC 
# MAGIC 在箱型图中，我们从上四分位数到下四分位数绘制一个盒子，然后用一条垂直触须（形象地称为“盒须”）穿过盒子的中间。上垂线延伸至上边缘（最大值），下垂线延伸至下边缘（最小值）。

# COMMAND ----------

# MAGIC %md
# MAGIC ## 提琴图
# MAGIC 小提琴图（Violin Plot）是用来展示数据分布状态以及概率密度的图表。这种图表结合了箱形图和密度图的特征。小提琴图跟箱形图类似，不同之处在于小提琴图还显示数据在不同数值下的概率密度。
# MAGIC 
# MAGIC 小提琴图使用核密度估计（KDE）来计算样本的分布情况，图中要素包括了中位数、四分位间距以及置信区间。在数据量非常大且不方便一一展示的时候，小提琴图特别适用。

# COMMAND ----------

# MAGIC %md
# MAGIC ## 更多案例
# MAGIC https://matplotlib.org/stable/gallery/index.html
