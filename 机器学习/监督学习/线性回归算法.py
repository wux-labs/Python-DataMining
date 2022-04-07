# Databricks notebook source
# MAGIC %md
# MAGIC # 基本函数如何确立

# COMMAND ----------

# MAGIC %md
# MAGIC ## 回归的形式——标记/label
# MAGIC 
# MAGIC * 回归模型可预测连续值。例如，回归模型做出的预测可回答如下问题：
# MAGIC   * 加利福尼亚州一栋房产的价值是多少？
# MAGIC   * 用户点击此广告的概率是多少？
# MAGIC * 分类模型可预测离散值。例如，分类模型做出的预测可回答如下问题：
# MAGIC   * 某个指定电子邮件是不是垃圾邮件？
# MAGIC   * 这是一张狗、猫还是仓鼠的照片？

# COMMAND ----------

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression # 线性回归

x = 2 * np.random.rand(100, 1)
# print("x = ", x)
y = 3 + 6 * x + np.random.rand(100,1)
# print("y = ", y)
lin_reg = LinearRegression()
lin_reg.fit(x, y)
lin_reg.coef_
print("0 = ", lin_reg.coef_)
print("截距项 = ", lin_reg.intercept_)
x_new = np.array([[0],[2]])
print(x_new)
y_predict = lin_reg.predict(x_new)
print(y_predict)

# COMMAND ----------

plt.plot(x_new, y_predict, 'r-')
plt.plot(x, y, 'k.')

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC 机器学习的所有算法，系统都是学的参数，我们不仅要代码实现出来，在学习算法的时候，也要进行“参数估计”。

# COMMAND ----------

# MAGIC %md
# MAGIC # 目标函数如何建立
