# Databricks notebook source
# MAGIC %md
# MAGIC # 预处理数据
# MAGIC 
# MAGIC sklearn.preprocessing软件包提供了几个常见的实用程序函数和转换器类，用于将原始特征向量更改为更适合下游估计器的表示形式。
# MAGIC 
# MAGIC 通常，学习算法受益于数据集的标准化。如果集合中存在一些异常值，则稳定的缩放或转换更合适。

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## 标准化，或均值去除和方差缩放
# MAGIC 
# MAGIC 数据集的 标准化 对scikit-learn中实现的大多数机器学习算法来说是 常见的要求 。如果个别特征或多或少看起来不是很像标准正态分布(具有零均值和单位方差)，那么它们的表现力可能会较差。
# MAGIC 
# MAGIC 在实际情况中,我们经常忽略特征的分布形状，直接经过**去均值**来对某个特征进行**中心化**，再通过除以非常量特征(non-constant features)的**标准差**进行缩放。
# MAGIC 
# MAGIC 例如，在机器学习算法的目标函数(例如SVM的RBF内核或线性模型的l1和l2正则化)，许多学习算法中目标函数的基础都是假设所有的特征都是零均值并且具有同一阶数上的方差。如果某个特征的方差比其他特征大几个数量级，那么它就会在学习算法中占据主导位置，导致学习器并不能像我们说期望的那样，从其他特征中学习。
# MAGIC 
# MAGIC preprocessing 模块提供了 StandardScaler 实用工具类，这是一种对类似数组的数据集执行以下操作的快速简便的方法：

# COMMAND ----------

from sklearn import preprocessing
import numpy as np

X_train = np.array([[ 1., -1.,  2.],
                    [ 2.,  0.,  0.],
                    [ 0.,  1., -1.]])

scaler = preprocessing.StandardScaler().fit(X_train)
# scaler 缩放器实例
# scaler.mean_ 均值
# scaler.scale_ 缩放比例（标准差）
scaler, scaler.mean_, scaler.scale_

# COMMAND ----------

mean = np.mean([1,2,0])
mean, np.sqrt(((1 - mean)**2 + (2 - mean)**2 + (0 - mean)**2)/3)

# COMMAND ----------

mean = np.mean([-1,0,1])
mean, np.sqrt(((-1 - mean)**2 + (0 - mean)**2 + (1 - mean)**2)/3)

# COMMAND ----------

mean = np.mean([2,0,-1])
mean, np.sqrt(((2 - mean)**2 + (0 - mean)**2 + (-1 - mean)**2)/3)

# COMMAND ----------

np.mean(X_train, axis=0), np.sqrt(np.var(X_train, axis=0))

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC 经过缩放后的数据具有零均值以及标准方差：

# COMMAND ----------

X_scaled = scaler.transform(X_train)
X_scaled, X_scaled.mean(axis=0), X_scaled.std(axis=0)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC StandardScaler 类实现 Transformer API 以计算训练集上的均值和标准差，以便以后能够在测试集上重新应用相同的转换。因此，此类适合在管道的早期步骤中使用：

# COMMAND ----------

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = make_classification(random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
pipe = make_pipeline(StandardScaler(), LogisticRegression())
pipe.fit(X_train, y_train)

pipe.score(X_test, y_test)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC 你也可以通过在构造函数 StandardScaler 中传入参数 with_mean=False 或者 with_std=False 来取消中心化或缩放操作。

# COMMAND ----------

# MAGIC %md
# MAGIC ### StandardScaler
# MAGIC 
# MAGIC > class sklearn.preprocessing.StandardScaler(*, copy=True, with_mean=True, with_std=True)
# MAGIC 
# MAGIC 通过移除均值并缩放到单位方差来标准化特征。
# MAGIC 
# MAGIC **标准化(Standardization)：** 当数据(x)按均值(μ)中心化后，再按标准差(σ)缩放，数据就会服从为均值为0，方差为1的正态分布（即标准正态分布），而这个过程，就叫做**数据标准化**。
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=x^*%3D%5Cfrac%7Bx_i-%5Cmu%7D%7B%5Csigma%7D)
# MAGIC 
# MAGIC 此缩放器还可以通过传递 with_mean=False 应用于稀疏 CSR 或 CSC 矩阵，以避免破坏数据的稀疏结构。

# COMMAND ----------

# MAGIC %md
# MAGIC #### 参数

# COMMAND ----------

# MAGIC %md
# MAGIC ##### copy
# MAGIC 
# MAGIC **bool, default=True**
# MAGIC 
# MAGIC 如果为 False，尝试避免复制，而是改为进行就地缩放。这不能保证总是在原地工作；例如，如果数据不是NumPy数组或scipy.sparse CSR矩阵，则仍可能返回副本。

# COMMAND ----------

# MAGIC %md
# MAGIC ##### with_mean
# MAGIC 
# MAGIC **bool, default=True**
# MAGIC 
# MAGIC 如果为 True，则在缩放之前将数据中心化。当尝试对稀疏矩阵进行时，这不起作用（并且会引发异常），因为将它们中心化需要构建一个密集的矩阵，在常见用例中，该矩阵可能太大而无法放入内存中。

# COMMAND ----------

# MAGIC %md
# MAGIC ##### with_std
# MAGIC 
# MAGIC **bool, default=True**
# MAGIC 
# MAGIC 如果为 True，则将数据缩放为单位方差（或等效地，单位标准差）。

# COMMAND ----------

# MAGIC %md
# MAGIC #### 属性

# COMMAND ----------

# MAGIC %md
# MAGIC ##### scale_
# MAGIC 
# MAGIC **ndarray of shape (n_features,) or None**
# MAGIC 
# MAGIC 对每个特征的数据进行相对缩放，以实现零均值和单位方差。通常，这是使用 np.sqrt(var_) 计算的。如果方差为零，则我们无法实现单位方差，并且数据保持原样，给出的比例因子为 1。 当 with_std=False，scale_ 等于 None。
# MAGIC 
# MAGIC > 版本 0.17 中的新功能：scale_

# COMMAND ----------

# MAGIC %md
# MAGIC ##### mean_
# MAGIC 
# MAGIC **ndarray of shape (n_features,) or None**
# MAGIC 
# MAGIC 训练集中每个特征的平均值。当 with_mean=False 等于 None。

# COMMAND ----------

# MAGIC %md
# MAGIC ##### var_
# MAGIC 
# MAGIC **ndarray of shape (n_features,) or None**
# MAGIC 
# MAGIC 训练集中每个特征的方差，用于计算 scale_。当 with_std=False 等于 None。

# COMMAND ----------

# MAGIC %md
# MAGIC ##### n_features_in_
# MAGIC 
# MAGIC **int**
# MAGIC 
# MAGIC 在拟合过程中用到的特征数量。
# MAGIC 
# MAGIC > 0.24 版中的新功能。

# COMMAND ----------

# MAGIC %md
# MAGIC ##### n_samples_seen_
# MAGIC 
# MAGIC **int or ndarray of shape (n_features,)**
# MAGIC 
# MAGIC 估计器为每个特征处理的样本数。如果没有缺少的样本，则 n_samples_seen 将是一个整数，否则它将是 dtype int 的数组。如果使用 sample_weights，它将是一个浮点数（如果没有丢失的数据）或一个dtype float数组，该数组对到目前为止看到的权重求和。
