# Databricks notebook source
# MAGIC %md
# MAGIC # 拟合正弦曲线

# COMMAND ----------

# MAGIC %md
# MAGIC ## 导入依赖

# COMMAND ----------

from sklearn import tree
from sklearn.model_selection import train_test_split, cross_val_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import graphviz

# COMMAND ----------

# MAGIC %md
# MAGIC ## 生成数据集

# COMMAND ----------

# 生成随机数种子
rng = np.random.RandomState(1)

# COMMAND ----------

# 获取样本数量（x轴数据）
X = np.arange(0,8,0.1)
X

# COMMAND ----------

# 获取正弦曲线
y = np.sin(X)
y

# COMMAND ----------

# 添加噪声
y[::5] += (0.5 - rng.rand(16))
y

# COMMAND ----------

plt.figure(figsize=(20,8))
plt.scatter(X, y)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 模型训练

# COMMAND ----------

regressor = tree.DecisionTreeRegressor(max_depth=2)

# 决策树只能接收2维数据，目前的X是1维的数据，所以不支持
# regressor.fit(X, y)

# 对X进行增维后进行训练
regressor.fit(X[:, np.newaxis], y)

# COMMAND ----------

regr_1 = tree.DecisionTreeRegressor(max_depth=2)
regr_2 = tree.DecisionTreeRegressor(max_depth=5)
regr_1.fit(X[:, np.newaxis], y)
regr_2.fit(X[:, np.newaxis], y)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 结果预测

# COMMAND ----------

# 准备测试集数据
X_test = np.arange(0, 8, 0.05)[:, np.newaxis]

# COMMAND ----------

y_test = regressor.predict(X_test)
y_test

# COMMAND ----------

y_1 = regr_1.predict(X_test)
y_2 = regr_2.predict(X_test)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 数据可视化

# COMMAND ----------

plt.figure(figsize=(20,8))
plt.scatter(X, y, label="data")
plt.plot(X_test, y_test)
plt.plot(X_test, y_1,label="max_depth=2")
plt.plot(X_test, y_2,label="max_depth=5")
plt.legend()

# COMMAND ----------

# MAGIC %md
# MAGIC 从上图结果看出，max_depth=2模型更好一些，max_depth=5的模型有过拟合的趋势。
