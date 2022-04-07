# Databricks notebook source
# MAGIC %md
# MAGIC # Matplotlib介绍
# MAGIC 内容预览  
# MAGIC * 什么是Matplotlib
# MAGIC * 为什么要学习Matplotlib
# MAGIC * 实现一个简单的Matplotlib画图
# MAGIC * 认识Matplotlib图像结构
# MAGIC * Matplotlib三层结构

# COMMAND ----------

# MAGIC %md
# MAGIC ## 什么是Matplotlib
# MAGIC * 专门用于开发2D图表
# MAGIC * 使用起来及其简单
# MAGIC * 以渐进、交互式方式实现数据可视化

# COMMAND ----------

# MAGIC %md
# MAGIC ## 为什么要学习Matplotlib
# MAGIC 可视化是在整个数据挖掘的关键辅助工具，可以清晰的理解数据，从而调整我们的分析方法
# MAGIC * 能将数据进行可视化，更直观的呈现
# MAGIC * 使数据更加客观、更具说服力

# COMMAND ----------

# MAGIC %md
# MAGIC ## 引入模块
# MAGIC 在正式使用Matplotlib之前，需要在代码中将其引入

# COMMAND ----------

# 引入 matplotlib.pyplot 模块
import matplotlib.pyplot as plt

# 引入其他需要的模块
import random

# COMMAND ----------

# MAGIC %md
# MAGIC ## 实现一个简单的Matplotlib画图

# COMMAND ----------

plt.figure()
plt.plot([1,0,8],[4,5,6])
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 认识Matplotlib图像结构

# COMMAND ----------

# MAGIC %md
# MAGIC ## Matplotlib三层结构

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1 容器层
# MAGIC 容器层主要由Canvas（画板层）、Figure（画布层）、Axes（绘图区/坐标系）组成。

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2 辅助显示层
# MAGIC 辅助显示层是Axes内的除了根据数据绘制出的图像以外的内容。主要包括：外观（facecolor）、边框线（spines）、坐标轴（axis）、坐标轴名称（axis label）、坐标轴刻度（tick）、坐标轴刻度标签（tick label）、网格线（grid）、图例（legend）、标题（title）等内容。

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3 图像层
# MAGIC 图像层指Axes内通过plot、scatter、bar、histogram、pie等函数根据数据绘制出的图像。

# COMMAND ----------

# MAGIC %md
# MAGIC # 折线图（plot）与基础绘图功能

# COMMAND ----------

# MAGIC %md
# MAGIC ## 折线图绘制与保存图片
# MAGIC 为了更好地理解所有基础绘图功能，我们通过天气温度变化的绘图来融合所有的基础API使用

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1 matplotlib.pyplot 模块
# MAGIC 包含了一系列类似于matlab的画图函数。

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2 折现绘制与显示
# MAGIC 展现一周的天气，比如从星期一到星期日的天气温度

# COMMAND ----------

# 1、创建画布（容器层）
plt.figure(figsize=(20,8))

# 2、绘制折线图（图像层）
plt.plot([1,2,3,4,5,6,7],[17,18,17,15,11,11,13])

# 3、显示图像
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC 使用`plt.savefig(path)`可以将图片保存到指定的路径。
# MAGIC > 注意：plt.show()会释放figure资源，如果在show()之后再去savefig(path)进行保存，只能保存空白图片。

# COMMAND ----------

# 设置画布大小
# figsize 指定画布大小，dpi 指定分辨率
plt.figure(figsize=(20,8),dpi=100)

# 准备数据
x = range(60)
y1 = [random.uniform(15,18) for i in x]
y2 = [random.uniform(5,8) for i in x]

# 绘制折线图
# 绘制第一条折线
plt.plot(x, y1,label="Plot 1")
# 绘制第二条折线
plt.plot(x, y2, color='r', linestyle="--",label="Plot 2")

# 修改x、y的刻度
# 准备x的刻度说明
x_label = ["11:{}".format(i) for i in x]
# 设置x的刻度范围(0-60)及步长(5)
plt.xticks(x[::5], x_label[::5])
# 设置y的刻度范围(0-30)及步长(5)
plt.yticks(range(0, 30, 5))

# 添加网格显示
plt.grid(True, linestyle="--",alpha=0.5)

# 添加坐标轴名称、标题
plt.xlabel("Time")
plt.ylabel("Temperature")
plt.title("Temperature Plot")

# 显示图例
# 对应于plt.plot(x, y1,label="Plot 1")中的 label="Plot 1"
plt.legend()

# 显示图像
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC 多个坐标系显示 - plt.subplots
# MAGIC * matplotlib.pyplot.subplots(nrows=1,ncols=1,**fig_kw)创建一个带有多个axes的图

# COMMAND ----------

# 创建具有两个绘图区的画布
figure, axes = plt.subplots(nrows=1,ncols=2,figsize=(20,8))

# 准备数据
x = range(60)
y1 = [random.uniform(15,18) for i in x]
y2 = [random.uniform(5,8) for i in x]

# 绘制折线图
# 绘制第一条折线
axes[0].plot(x, y1,label="Plot 1")
# 绘制第二条折线
axes[1].plot(x, y2, color='r', linestyle="--",label="Plot 2")

# 修改x、y的刻度
# 准备x的刻度说明
x_label = ["11:{}".format(i) for i in x]
# 设置x的刻度范围(0-60)及步长(5)
axes[0].set_xticks(x[::5])
axes[0].set_xticklabels( x_label[::5])
axes[1].set_xticks(x[::5])
axes[1].set_xticklabels( x_label[::5])
# 设置y的刻度范围(0-30)及步长(5)
axes[0].set_yticks(range(0, 30, 5))
axes[1].set_yticks(range(0, 30, 5))

# 添加网格显示
axes[0].grid(True, linestyle="--",alpha=0.5)
axes[1].grid(True, linestyle="--",alpha=0.5)

# 添加坐标轴名称、标题
axes[0].set_xlabel("Time")
axes[1].set_xlabel("Time")
axes[0].set_ylabel("Temperature")
axes[1].set_ylabel("Temperature")
axes[0].set_title("Temperature Plot")
axes[1].set_title("Temperature Plot")

# 显示图例
# 对应于plt.plot(x, y1,label="Plot 1")中的 label="Plot 1"
axes[0].legend()
axes[1].legend()

# 显示图像
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC # 散点图（scatter）
# MAGIC 散点图是指在回归分析中，数据点在直角坐标系平面上的分布图，散点图表示因变量随自变量而变化的大致趋势，据此可以选择合适的函数对数据点进行拟合。  
# MAGIC 用两组数据构成多个坐标点，考察坐标点的分布，判断两变量之间是否存在某种关联或总结坐标点的分布模式。散点图将序列显示为一组点。值由点在图表中的位置表示。类别由图表中的不同标记表示。散点图通常用于比较跨类别的聚合数据。

# COMMAND ----------

# 1、创建画布（容器层）
plt.figure(figsize=(20,8))

# 2、绘制折线图（图像层）
plt.scatter(range(50),[random.uniform(1,20) for i in range(50)])

# 3、显示图像
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC # 柱状图（bar）

# COMMAND ----------

# 1、创建画布（容器层）
plt.figure(figsize=(20,8))

# 2、绘制折线图（图像层）
plt.bar(range(50),[random.uniform(1,20) for i in range(50)])

# 3、显示图像
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC # 直方图（histogram）

# COMMAND ----------

# 1、创建画布（容器层）
plt.figure(figsize=(20,8))

# 2、绘制折线图（图像层）
plt.hist([random.uniform(1,20) for i in range(50)])

# 3、显示图像
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC # 饼图（pie）

# COMMAND ----------

# 1、创建画布（容器层）
plt.figure(figsize=(20,8))

list = [random.uniform(1,5) for i in range(10)]

# 2、绘制折线图（图像层）
plt.pie(list)

# 3、显示图像
plt.show()
