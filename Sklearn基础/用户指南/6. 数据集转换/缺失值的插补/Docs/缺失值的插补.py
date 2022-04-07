# Databricks notebook source
# MAGIC %md
# MAGIC # 缺失值的插补
# MAGIC 
# MAGIC 由于各种原因，许多现实世界的数据集都包含缺失值，通常编码为空白、NaN 或其他占位符。然而，这样的数据集与sklearn估计器不兼容，后者假设数组中的所有值都是数字值，并且都具有并具有意义。使用不完整数据集的基本策略是丢弃整行和/或包含缺失值的列。但是，这是以丢失数据为代价的，这些数据可能是有价值的（即使不完整）。更好的策略是插补缺失值，即从数据的已知部分推断它们。

# COMMAND ----------

# MAGIC %md
# MAGIC ## 单变量与多变量插补
# MAGIC 
# MAGIC 一种类型的插补算法是单变量，它仅使用该特征维度中的非缺失值（例如impute.SimpleImputer）来插补第 i 个特征维度中的值。相比之下，多元插补算法使用整组可用的特征维度来估计缺失值（例如impute.IterativeImputer）。

# COMMAND ----------

# MAGIC %md
# MAGIC ## 单变量特征插补
# MAGIC 
# MAGIC SimpleImputer类提供了用于插补缺失值的基本策略。可以使用提供的常量值估算缺失值，也可以使用缺失值所在的每列的统计信息（平均值、中位数或最频繁值）进行估算。此类还允许使用不同的缺失值编码。

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC 以下代码段演示如何使用包含缺失值的列（轴 0）的平均值替换编码为np.nan的缺失值：

# COMMAND ----------

import numpy as np
from sklearn.impute import SimpleImputer
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
imp.fit([[1, 2], [np.nan, 3], [7, 6]])

X = [[np.nan, 2], [6, np.nan], [7, 6]]
print(imp.transform(X))

# COMMAND ----------

# MAGIC %md
# MAGIC ### SimpleImputer
# MAGIC 
# MAGIC > class sklearn.impute.SimpleImputer(*, missing_values=nan, strategy='mean', fill_value=None, verbose=0, copy=True, add_indicator=False)
# MAGIC 
# MAGIC 用于完成缺失值的插补变压器。
# MAGIC 
# MAGIC > 版本 0.20 中的新功能：SimpleImputer替换了之前已删除的sklearn.preprocessing.Imputer估计器。

# COMMAND ----------

# MAGIC %md
# MAGIC #### 参数

# COMMAND ----------

# MAGIC %md
# MAGIC ##### missing_values
# MAGIC 
# MAGIC **int, float, str, np.nan or None, default=np.nan**
# MAGIC 
# MAGIC 缺失值的占位符。默认空值np.nan。

# COMMAND ----------

# MAGIC %md
# MAGIC ##### strategy
# MAGIC 
# MAGIC **string, default='mean'**
# MAGIC 
# MAGIC 插补策略。
# MAGIC 
# MAGIC * 如果为"mean"，则使用每列的均值替换缺失值。只能与数值数据一起使用。
# MAGIC * 如果为"median"，则使用每列的中位数替换缺失值。只能与数值数据一起使用。
# MAGIC * 如果为"most_frequent"，则使用每列中最常见的值替换缺失值。可与字符串或数值数据一起使用。如果有多个此类值，则仅返回最小的值。
# MAGIC * 如果为"constant"，则将缺失值替换为fill_value。可与字符串或数值数据一起使用。
# MAGIC 
# MAGIC > 0.20 版中的新功能：用于固定值插补的strategy="constant"。

# COMMAND ----------

# MAGIC %md
# MAGIC ##### fill_value
# MAGIC 
# MAGIC **string or numerical value, default=None**
# MAGIC 
# MAGIC 当strategy="constant"时，fill_value用于替换所有出现的missing_values。如果保留默认值，则在插补数值数据时，fill_value将为 0，字符串或对象数据类型为"missing_value"。

# COMMAND ----------

# MAGIC %md
# MAGIC ##### verbose
# MAGIC 
# MAGIC **integer, default=0**
# MAGIC 
# MAGIC 控制插补的详细程度。

# COMMAND ----------

# MAGIC %md
# MAGIC ##### copy
# MAGIC 
# MAGIC **boolean, default=True**
# MAGIC 
# MAGIC 如果为 True，则将创建 X 的副本。如果为 False，则将尽可能就地进行插补。请注意，在以下情况下，将始终制作新副本，即使：copy=False
# MAGIC * 如果 X 不是浮点值数组;
# MAGIC * 如果 X 被编码为 CSR 矩阵;
# MAGIC * 如果 add_indicator=True。

# COMMAND ----------

# MAGIC %md
# MAGIC ##### add_indicator
# MAGIC 
# MAGIC **boolean, default=False**
# MAGIC 
# MAGIC 如果为 True，则 MissingIndicator 变换将堆叠到插值变换的输出上。这允许预测估计器在估算的情况下解释缺失。如果要素在适合/训练时没有缺失值，则即使转换/测试时缺少值，该要素也不会显示在缺失指示器上。

# COMMAND ----------

# MAGIC %md
# MAGIC #### 属性

# COMMAND ----------

# MAGIC %md
# MAGIC ##### statistics_
# MAGIC 
# MAGIC **array of shape (n_features,)**
# MAGIC 
# MAGIC 每个要素的插补填充值。

# COMMAND ----------

# MAGIC %md
# MAGIC ##### indicator_
# MAGIC 
# MAGIC **MissingIndicator**
# MAGIC 
# MAGIC 用于为缺失值添加二进制指标的指标。如果add_indicator=False则为None。
