# Databricks notebook source
# MAGIC %md
# MAGIC # 红酒数据集

# COMMAND ----------

# MAGIC %md
# MAGIC ## 导入依赖

# COMMAND ----------

from sklearn import tree
from sklearn import ensemble
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, cross_val_score
import pandas as pd
import matplotlib.pyplot as plt
import graphviz

# COMMAND ----------

# MAGIC %md
# MAGIC ## 读取数据
# MAGIC 
# MAGIC 读取数据集，并观察数据。

# COMMAND ----------

# 加载红酒数据集
wine = load_wine()

# COMMAND ----------

# 这是一个数据字典结构的数据
type(wine), wine

# COMMAND ----------

# MAGIC %md
# MAGIC ## 数据划分
# MAGIC 
# MAGIC 将数据划分成训练集和测试集，观察一下划分情况。

# COMMAND ----------

x_train, x_test, y_train, y_test = train_test_split(wine.data, wine.target, test_size=0.3)

# COMMAND ----------

x_train, x_train.shape, y_train, y_train.shape

# COMMAND ----------

# MAGIC %md
# MAGIC ## 模型训练
# MAGIC 
# MAGIC 训练模型并验证。

# COMMAND ----------

dtc = tree.DecisionTreeClassifier(random_state=0)
rfc = ensemble.RandomForestClassifier(random_state=0)

dtc = dtc.fit(x_train, y_train)
rfc = rfc.fit(x_train, y_train)

# 返回预测的准确度 accuracy
score_dtc = dtc.score(x_test, y_test)
score_rfc = rfc.score(x_test, y_test)

score_dtc, score_rfc

# COMMAND ----------

# 查看森林中树的状况
# 默认情况下森林中有100棵树，每棵树的random_state都不一样，所以是100棵不同的树
# 由于森林中每棵树都不一样，所以随机森林取平均值或少数服从多数后，效果比单棵分类树的效果好
rfc.estimators_

# COMMAND ----------

# MAGIC %md
# MAGIC ### 交叉验证
# MAGIC 
# MAGIC 从上面的结果我们可以看到，随机森林的效果要比单棵树的效果好。
# MAGIC 
# MAGIC 下面用交叉验证来验证下。

# COMMAND ----------

dtc = tree.DecisionTreeClassifier(random_state=0)
dtc_cvs = cross_val_score(dtc, wine.data, wine.target, cv=10)

rfc = ensemble.RandomForestClassifier(random_state=0, n_estimators=25)
rfc_cvs = cross_val_score(rfc, wine.data, wine.target, cv=10)

plt.plot(range(1,11), dtc_cvs, label="决策树")
plt.plot(range(1,11), rfc_cvs, label="随机森林")

plt.legend()


# COMMAND ----------

# MAGIC %md
# MAGIC ### 多次交叉验证

# COMMAND ----------

dtc_scores = []
rfc_scores = []

for i in range(10):
    dtc = tree.DecisionTreeClassifier()
    dtc_scores.append(cross_val_score(dtc, wine.data, wine.target, cv=10).mean())

    rfc = ensemble.RandomForestClassifier()
    rfc_scores.append(cross_val_score(rfc, wine.data, wine.target, cv=10).mean())

plt.figure(figsize=(20,8))
plt.plot(range(1,11), dtc_scores, label="决策树")
plt.plot(range(1,11), rfc_scores, label="随机森林")

plt.legend()


# COMMAND ----------

# MAGIC %md
# MAGIC ## 参数调整

# COMMAND ----------

# MAGIC %md
# MAGIC ### 树的数量的学习曲线
# MAGIC 
# MAGIC 任何模型都有决策边界，n_estimators达到一定的程度之后，随机森林的精确性往往不再上升，或开始波动。

# COMMAND ----------

scores = []
for i in range(80):    
    rfc = ensemble.RandomForestClassifier(random_state=0, n_estimators=i+1)
    scores.append(cross_val_score(rfc, wine.data, wine.target, cv=10).mean())

plt.figure(figsize=(20,8))
plt.plot(range(80), scores)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 使用袋外数据测试

# COMMAND ----------

rfc = ensemble.RandomForestClassifier(random_state=0, oob_score=True)
rfc = rfc.fit(wine.data, wine.target)

# COMMAND ----------

rfc.oob_score_
