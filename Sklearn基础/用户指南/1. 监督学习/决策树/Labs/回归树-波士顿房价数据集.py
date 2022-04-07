# Databricks notebook source
# MAGIC %md
# MAGIC # 波士顿房价数据集

# COMMAND ----------

# MAGIC %md
# MAGIC ## 导入依赖

# COMMAND ----------

from sklearn import tree
from sklearn.datasets import load_boston
from sklearn.model_selection import cross_val_score

# COMMAND ----------

# MAGIC %md
# MAGIC ## 读取数据
# MAGIC 
# MAGIC 读取数据集，并观察数据。

# COMMAND ----------

boston = load_boston()

# COMMAND ----------

print(type(boston))

# 这是一个数据字典结构的数据
boston

# COMMAND ----------

# MAGIC %md
# MAGIC ## 模型训练
# MAGIC 
# MAGIC 训练模型并验证。
# MAGIC 
# MAGIC 交叉验证会自动划分数据，所以不需要手动划分数据的训练集和测试集。

# COMMAND ----------

regressor = tree.DecisionTreeRegressor(random_state=0) # 实例化

# COMMAND ----------

# score默认返回 R平方，越接近1越好
cross_val_score(regressor, boston.data, boston.target, cv=10)

# COMMAND ----------

# 指定score返回均方误差，越小越好
cross_val_score(regressor, boston.data, boston.target, cv=10, scoring="neg_mean_squared_error")
