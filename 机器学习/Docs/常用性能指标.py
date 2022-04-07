# Databricks notebook source
# MAGIC %md
# MAGIC # 常用性能指标
# MAGIC 
# MAGIC 在机器学习中，性能指标(Metrics)是衡量一个模型好坏的关键，通过衡量模型输出y_predict 和 y_true之间的某种"距离"得出的。
# MAGIC 
# MAGIC 性能指标往往是我们做模型时的最终目标，如**准确率**，**召回率**，**敏感度**等等，但是性能指标常常因为不可微分，无法作为优化的loss函数，因此采用如cross-entropy, rmse等“距离”可微函数作为优化目标，以期待在loss函数降低的时候，能够提高性能指标。而最终目标的性能指标则作为模型训练过程中，作为验证集做决定(early stoping或model selection)的主要依据，与训练结束后评估本次训练出的模型好坏的重要标准。

# COMMAND ----------

# sklearn当中的模型评估指标（打分）列表
import sklearn
sorted(sklearn.metrics.SCORERS.keys())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 准确率（Accuracy）
