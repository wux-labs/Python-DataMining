# Databricks notebook source
# MAGIC %md
# MAGIC # Matplotlib Figure
# MAGIC 在 Matplotlib 中，面向对象编程的核心思想是创建图形对象（figure object）。通过图形对象来调用其它的方法和属性，这样有助于我们更好地处理多个画布。在这个过程中，pyplot 负责生成图形对象，并通过该对象来添加一个或多个 axes 对象（即绘图区域）。
# MAGIC 
# MAGIC Matplotlib 提供了matplotlib.figure图形类模块，它包含了创建图形对象的方法。通过调用 pyplot 模块中 figure() 函数来实例化 figure 对象。

# COMMAND ----------

from matplotlib import pyplot as plt
#创建图形对象
figure = plt.figure()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 参数说明
# MAGIC | 参数 | 说明 |
# MAGIC | ----------- | ----------- |
# MAGIC | figsize | 指定画布的大小，(宽度,高度)，单位为英寸。 |
# MAGIC | dpi | 指定绘图对象的分辨率，即每英寸多少个像素，默认值为80。 |
# MAGIC | facecolor | 背景颜色。 |
# MAGIC | edgecolor | 边框颜色。 |
# MAGIC | frameon | 是否显示边框。 |

# COMMAND ----------

# MAGIC %md
# MAGIC ## 案例

# COMMAND ----------

# MAGIC %md
# MAGIC ### 创建一个空白画布
# MAGIC 下面的例子创建了一个空白画布。
# MAGIC 
# MAGIC > 为了能使画布显示出来，在画布上添加了坐标系。

# COMMAND ----------

figure=plt.figure()
ax=figure.add_axes([0,0,1,1])

plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 指定画布的大小
# MAGIC 以下例子设置了画布的大小。

# COMMAND ----------

figure=plt.figure(figsize=(20,8))
ax=figure.add_axes([0,0,1,1])

plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 设置画布的背景色

# COMMAND ----------

figure=plt.figure(figsize=(20,8), facecolor='r')
ax=figure.add_axes([0,0,1,1])

plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 设置画布的边框颜色

# COMMAND ----------

figure=plt.figure(figsize=(20,8), edgecolor='r', frameon=True)
ax=figure.add_axes([0,0,1,1])

plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 设置画布是否显示边框

# COMMAND ----------

figure=plt.figure(figsize=(20,8), facecolor='r', frameon=False)
ax=figure.add_axes([0,0,1,1])

plt.show()
