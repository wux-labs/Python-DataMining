# Databricks notebook source
# MAGIC %md
# MAGIC # 填补缺失值
# MAGIC 
# MAGIC 我们从显示中收集的数据，几乎不可能是完美无缺的，往往都会有一些缺失值。面对缺失值，我们可以直接将含有缺失值的样本删除，或这将含有缺失值的特征删除，这是一种有效的方法。但有时候填补缺失值会比直接丢弃样本、特征效果更好，即使我们其实并不知道缺失值的真实样子。
# MAGIC 
# MAGIC 在sklearn中，我们可以使用sklearn.SimpleImputer来轻松地将均值、中位数、众数，或者其他最常用的数值填补到数据中。
# MAGIC 
# MAGIC 在这个案例中，我们使用均值、常数，和随机森林回归来填补缺失值，并验证这几种状态下的拟合情况，找出对使用的数据集来说最佳的缺失值填补方法。

# COMMAND ----------

# MAGIC %md
# MAGIC ## 导入依赖

# COMMAND ----------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_boston
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import cross_val_score

# COMMAND ----------

# MAGIC %md
# MAGIC ## 导入数据集

# COMMAND ----------

boston = load_boston()

print(boston.data.shape)

X_full, y_full = boston.data, boston.target

n_samples = X_full.shape[0]
n_features = X_full.shape[1]

# COMMAND ----------

# MAGIC %md
# MAGIC ## 为完整数据集放入缺失值
# MAGIC 
# MAGIC 首先确定我们希望放入的缺失值数据的比例。

# COMMAND ----------

rng = np.random.RandomState(0)

# 假如缺失数据的比例是 50%
missing_rate = 0.5

# np.floor 向下取整，返回 .0 格式的浮点数
# 所有数据要随机遍布在数据集的各行各列当中，而一个缺失的数据会需要一个行索引和一个列索引
n_missing_samples = int(np.floor(n_samples * n_features * missing_rate))

missing_features = rng.randint(0,n_features,n_missing_samples)
missing_samples = rng.randint(0,n_samples,n_missing_samples)

# COMMAND ----------

missing_features, missing_samples

# COMMAND ----------

len(missing_features), len(missing_samples)

# COMMAND ----------

# 我们现在采样了3289个数据，远远超过我们的样本量506，所以我们使用随机抽取的函数randint。但如果而我们需要的数据量小于我们的样本506，那我们可以采用np.random.choice来抽样，choice会随机抽取不重复的随机数，因此可以帮助我们让数据更加分散，确保数据不会集中在一些行中。
# missing_samples = rng.choice(n_samples, n_missing_samples, replace=False)

X_missing = X_full.copy()
y_missing = y_full.copy()

X_missing[missing_samples, missing_features] = np.nan

# 转换成DataFrame是为了后续方便各种操作，numpy对矩阵的运算速度很快，但在索引等功能上却不如pandas
X_missing = pd.DataFrame(X_missing)

X_missing

# COMMAND ----------

# MAGIC %md
# MAGIC ## 使用0和均值填补缺失值

# COMMAND ----------

# strategy='mean'， 均值
# strategy='median'，中位数
# strategy='most_frequent'，众数
# strategy='constant'，常数

imp_0 = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)
X_missing_0 = imp_0.fit_transform(X_missing)

imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
X_missing_mean = imp_mean.fit_transform(X_missing)

# COMMAND ----------

pd.DataFrame(X_missing_mean).isnull().sum()
# 布尔值，False = 0， True = 1

# COMMAND ----------

# MAGIC %md
# MAGIC ## 使用随机森林填补缺失值
# MAGIC 
# MAGIC 任何回归都是从特征矩阵中学习，然后求解连续型标签y的过程，之所以能够实现这个过程，是因为回归算法认为特征矩阵和标签之间存在着某种联系。实际上，标签和特征是可以相互转换的，比如说，在一个用：“地区”、“环境”、“附件学校数量”，来预测“房价”的问题中，我们既可以用“地区”、“环境”、“附近学校数量”的数据来预测“房价”，也可以反过来，使用“环境”、“附近的学校数量”、“房价”的数据来预测“地区”。而回归填补缺失值，正是利用了这种思想。
# MAGIC 
# MAGIC 对于一个有 n 个特征的数据来说，其中特征 T 有缺失值，我们就把特征 T 当作标签，其他的 n-1 个特征和原本的标签组成新的特征矩阵。对于 T 来说，它没有缺失的部分就是我们的 y_train，这部分数据既有标签也有特征，而它缺失的部分，只有特征没有标签，就是我们需要预测的部分。
# MAGIC 
# MAGIC 特征 T 不缺失的值对应的其他 n-1 个特征 + 本来的标签：X_train  
# MAGIC 特征 T 不缺失的值：y_train
# MAGIC 
# MAGIC 特征 T 缺失的值对应的其他 n-1 个特征 + 本来的标签：X_test  
# MAGIC 特征 T 缺失的值：未知，我们需要预测的 y_test
# MAGIC 
# MAGIC 这种做法，对于某一个特征大量缺失，其他特征却很完整的情况，非常适用
# MAGIC 
# MAGIC 那如果数据中除了特征 T 之外，其他特征也有缺失值该怎么办？  
# MAGIC 遍历所有的特征，从缺失最少的开始进行填补（因为填补缺失最少的特征所需要的准确信息最少）。填补一个特征时，先将其他特征的缺失值用0代替，每完成一次回归预测，将将预测值放到原本的特征矩阵中，再继续填补下一个特征。每一次填补完毕，有缺失值的特征会减少一个，所以每次循环后，需要用0来填补的特征就越来越少。当进行到最后一个特征时（这个特征应该是所有特征中缺失值最多的），已经没有任何其他特征需要用0来进行填补了，而我们已经使用回归为其他特征填补了大量有效信息，可以用来填补缺失值最多的特征。
# MAGIC 
# MAGIC 遍历所有的特征后，数据就完整，不再有缺失值了。

# COMMAND ----------

X_missing_reg = X_missing.copy()

# COMMAND ----------

X_missing_reg

# COMMAND ----------

# 找出数据集中，缺失值从小到大排列的特征的顺序
sortindex = np.argsort(X_missing_reg.isnull().sum(axis=0)).values

for i in sortindex:
    # 构建我们的新特征矩阵和新标签
    df = X_missing_reg.copy()
    fillc = df.iloc[:, i]
    df = pd.concat([df.iloc[:, df.columns != i], pd.DataFrame(y_full)], axis=1)
    
    # 在新特征矩阵中，对含有缺失值的列，进行 0 的填补    
    df_0 = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0).fit_transform(df)
    
    # 找出我们的训练集和测试集
    # 要填充的特征 T 中不缺失的部分
    y_train = fillc[fillc.notnull()]
    # 要填充的特征 T 中缺失的部分
    y_test = fillc[fillc.isnull()]
    # 要填充的特征 T 中不缺失的部分
    X_train = df_0[y_train.index, :]
    # 要填充的特征 T 中缺失的部分
    X_test = df_0[y_test.index, :]
    
    # 用随机森林回归来填补缺失值
    rfc = RandomForestRegressor(n_estimators=100)
    rfc = rfc.fit(X_train, y_train)
    y_predict = rfc.predict(X_test)
    
    # 将填补好的特征返回到我们的原始特征矩阵中
    X_missing_reg.loc[X_missing_reg.iloc[:, i].isnull(), i] = y_predict

# COMMAND ----------

X_missing_reg

# COMMAND ----------

# MAGIC %md
# MAGIC ## 四个数据集的预测结果

# COMMAND ----------

X = [X_full, X_missing_0, X_missing_mean, X_missing_reg]

# 使用均方误差进行比较
mse = []

for x in X:
    estimator = RandomForestRegressor(random_state=0)
    # 使用均方误差
    scores = cross_val_score(estimator, x, y_full, scoring="neg_mean_squared_error", cv=5).mean()
    mse.append(scores * -1)

# 均方误差越小越好
mse

# COMMAND ----------

labels = ['原始数据', '使用0进行填充', '使用均值进行填充', '使用随机森林进行填充']

colors = ['r','g','b','orange']

plt.figure(figsize=(12,6))
ax = plt.subplot(111)

for i in np.arange(len(mse)):
    ax.barh(i, mse[i], color=colors[i])

ax.set_title('波士顿房价数据集缺失值填补的表现')

ax.set_xlabel("均方误差")
# ax.set_xlim(left=np.min(mse)*0.9, right=np.max(mse)*1.1)

ax.invert_yaxis()
ax.set_yticks(np.arange(len(mse)))
ax.set_yticklabels(labels)

