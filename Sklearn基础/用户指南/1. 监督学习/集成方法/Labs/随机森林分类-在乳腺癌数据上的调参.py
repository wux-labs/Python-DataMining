# Databricks notebook source
# MAGIC %md
# MAGIC # 在乳腺癌数据上的调参

# COMMAND ----------

# MAGIC %md
# MAGIC ## 导入依赖

# COMMAND ----------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import tree
from sklearn import ensemble
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import GridSearchCV, cross_val_score

# COMMAND ----------

# MAGIC %md
# MAGIC ## 读取数据
# MAGIC 
# MAGIC 读取数据集，并观察数据。

# COMMAND ----------

# 加载乳腺癌数据集
cancer = load_breast_cancer()

# COMMAND ----------

cancer, cancer.data, cancer.target, cancer.data.shape

# 可以看到，乳腺癌数据集有569条记录，30个特征，单看维度虽然不算太高，但是样本量非常少，很可能存在过拟合的情况。

# COMMAND ----------

# MAGIC %md
# MAGIC ## 模型训练
# MAGIC 
# MAGIC 训练模型并验证。

# COMMAND ----------

random_state=0

# COMMAND ----------

rfc = ensemble.RandomForestClassifier(random_state=random_state)
rfc_cvs = cross_val_score(rfc, cancer.data, cancer.target, cv=10).mean()
rfc_cvs

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC 从上面的结果可以看出，随机森林分类器在该数据集上天生具有很好的表现。

# COMMAND ----------

# MAGIC %md
# MAGIC ## 参数调优

# COMMAND ----------

# MAGIC %md
# MAGIC ### n_estimators 的学习曲线
# MAGIC 
# MAGIC 要看见n_estimators在什么取值开始变得平稳，是否一直推动模型整体准确率的上升等信息。
# MAGIC 
# MAGIC 第一次的学习曲线，可以先用来帮助我们划定范围，我们取每十个数作为一个阶段，来观察n_estimators的变化如何引起模型整体准确率的变化。

# COMMAND ----------

score1 = []

min_estimator=1
max_estimator=201
step=5
range1 = range(min_estimator,max_estimator,step)

for i in range1:
    rfc = ensemble.RandomForestClassifier(n_estimators=i, random_state=random_state)
    score = cross_val_score(rfc, cancer.data, cancer.target, cv=10).mean()
    score1.append(score)

plt.figure(figsize=(20,8))
plt.plot(range1, score1)

max_score = max(score1)
n_estimators = score1.index(max(score1)) * step + min_estimator

max_score, n_estimators

# COMMAND ----------

# MAGIC %md
# MAGIC ### 细化学习曲线
# MAGIC 
# MAGIC 上面的学习曲线可以大致确定模型效果较好时，n_estimators 的大概的取值范围，接下来我们就可以在确定好的范围内细化学习曲线。

# COMMAND ----------

score2 = []

range2 = range(n_estimators - 5,n_estimators + 5,1)

for i in range2:
    rfc = ensemble.RandomForestClassifier(n_estimators=i, random_state=random_state)
    score = cross_val_score(rfc, cancer.data, cancer.target, cv=10).mean()
    score2.append(score)

plt.figure(figsize=(20,8))
plt.plot(range2, score2)

max_score = max(score2)
n_estimators = score2.index(max(score2)) + n_estimators - 5

max_score, n_estimators

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC 调整n_estimators，模型的准确率上升了。
# MAGIC 
# MAGIC 接下来就进入网格搜索，我们使用网格搜索对参数一个个进行调整。
# MAGIC 
# MAGIC 为什么我们不同时调整多个参数呢？
# MAGIC * 同时调整多个参数会运行非常缓慢
# MAGIC * 同时调整多个参数会让我们无法理解参数的组合是怎么得来的，所以即便网格搜索调出来的效果不好，我们也不知道从哪里去改

# COMMAND ----------

# MAGIC %md
# MAGIC ### 网格搜索
# MAGIC 
# MAGIC 有一些参数是没有参照的，很难说清一个范围，这种情况下我们使用学习曲线，看趋势从曲线跑出的结果中选取一个更小的区间，再跑曲线。
# MAGIC 
# MAGIC * param_grid = {'n_estimaters': np.arange(0,200,10)}
# MAGIC * param_grid = {'max_depth': np.arange(1,20,1)}
# MAGIC * param_grid = {'max_leaf_nodes': np.arange(25,50,1)}
# MAGIC > 对于大型数据集，可以尝试从1000来构建，每100个叶子一个区间，再逐渐缩小范围
# MAGIC 
# MAGIC 有一些参数是可以找到一个范围的，或者说我们知道他们的取值，和随着他们的取值变化模型的整体准确率会如何变化，这样的参数我们就可以直接跑网格搜索
# MAGIC * param_grid = {'criterion': ['gini', 'entropy']}
# MAGIC * param_grid = {'min_samples_split': np.arange(2,2+20,1)}
# MAGIC * param_grid = {'min_samples_leaf': np.arange(1,1+10,1)}
# MAGIC * param_grid = {'max_features': np.arange(5,30,1)}

# COMMAND ----------

# MAGIC %md
# MAGIC #### max_depth
# MAGIC 
# MAGIC 按照参数对模型整体准确率的影响程度进行调参，首先调整max_depth。
# MAGIC 
# MAGIC 限制max_depth，是让模型变得简单。
# MAGIC 
# MAGIC 一般根据数据的大小来进行一个试探，乳腺癌数据很小，所以采样1\~10，或者1\~20这样的试探。但对于像digit recognition那样的大型数据来说，我们应该尝试30~50层的深度（或许还不足够）更应该画出学习曲线，来观察深度对模型的影响。

# COMMAND ----------

param_grid = {'max_depth': np.arange(1,20,1)}

rfc = ensemble.RandomForestClassifier(n_estimators=n_estimators,random_state=random_state)
gs = GridSearchCV(rfc,param_grid,cv=10)
gs.fit(cancer.data,cancer.target)

# best_params_ 最佳参数组合
# best_score_ 最佳参数组合下的最佳准确率
print(gs.best_params_, gs.best_score_)

max_depth = gs.best_params_['max_depth']
max_depth

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC 通常来说，随机森林应该在泛化误差最低点的右边，树模型应该倾向于过拟合，而不是欠拟合。
# MAGIC 
# MAGIC max_depth参数调整后有几种可能性：
# MAGIC * 将max_depth设置为有限之后，模型的准确率没有变化。
# MAGIC > 有可能是我们调整的n_estimators对于数据集来说太大，因此将模型拉到泛化误差最低点去了。
# MAGIC * 将max_depth设置为有限之后，模型的准确率下降了。
# MAGIC > 限制max_depth，是让模型变得简单，而模型整体的准确率下降了，即整体的泛化误差上升了，这说明模型现在位于泛化误差最低点的左边（偏差为主导的一边）。
# MAGIC 
# MAGIC 以上几种情况，都说明我们调整的n_estimators将模型拉到最接近泛化误差最低点去了，既然我们追求最低泛化误差，那我们就保留这个n_estimators，除非有其他的因素，可以帮助我们达到更高的准确率。
# MAGIC 
# MAGIC 当模型位于泛化误差最低点的左边时，我们需要的是增加模型复杂度（增加方差，减少偏差）的选项，因此max_depth应该尽量大，min_samples_leaf和min_samples_split都应该尽量小。这几乎是在说明，除了max_features，我们没有任何参数可以调整了，因为max_depth，min_samples_leaf和min_samples_split是剪枝参数，是减小复杂度的参数。
# MAGIC 
# MAGIC 在这里，我们可以预言，我们已经非常接近模型的上限，模型很可能没有办法再进步了。
# MAGIC 
# MAGIC 那我们这就来调整一下max_features，看看模型如何变化。

# COMMAND ----------

# MAGIC %md
# MAGIC #### max_features
# MAGIC 
# MAGIC max_features是唯一的既能够将模型往左（低方差高偏差）推，也能够将模型往右（高方差低偏差）推的参数。我们需要根据调参前，模型所在的位置（在泛化误差最低点的左边还是右边）来决定我们要将max_features往哪边调。
# MAGIC 
# MAGIC * 模型位于泛化误差最低点的左侧，我们需要的是更高的复杂度，因此我们应该把max_features往更大的方向调整，可用的特征越多，模型才会越复杂。max_features的默认值是sqrt(n_features)，因此我们使用这个值作为调参范围的最小值。
# MAGIC * 模型位于泛化误差最低点的右侧，我们需要的是更低的复杂度，因此我们应该把max_features往更小的方向调整，可用的特征越少，模型才会越简单。max_features的默认值是sqrt(n_features)，因此我们使用这个值作为调参范围的最大值。

# COMMAND ----------

param_grid = {'max_features':np.arange(5,30,1)}

rfc = ensemble.RandomForestClassifier(n_estimators=n_estimators,random_state=random_state, max_depth=max_depth)
gs = GridSearchCV(rfc,param_grid,cv=10)
gs.fit(cancer.data,cancer.target)

# best_params_ 最佳参数组合
# best_score_ 最佳参数组合下的最佳准确率
print(gs.best_params_, gs.best_score_)

max_features = gs.best_params_['max_features']
max_features

# COMMAND ----------

# MAGIC %md
# MAGIC max_features的默认最小值是sqrt(n_features) = sqrt(30) = 5，网格搜索返回了max_features = 5，可见max_features升高之后，模型的准确率可能会变低。这说明，我们把模型往右推，模型的泛化误差增加了。前面用max_depth往左推，泛化误差增加，现在用max_features往右推，泛化误差增加，这说明模型现在位于泛化误差最低点。

# COMMAND ----------

param_grid = {'max_features':np.arange(1,6,1)}

rfc = ensemble.RandomForestClassifier(n_estimators=n_estimators,random_state=random_state, max_depth=max_depth)
gs = GridSearchCV(rfc,param_grid,cv=10)
gs.fit(cancer.data,cancer.target)

# best_params_ 最佳参数组合
# best_score_ 最佳参数组合下的最佳准确率
print(gs.best_params_, gs.best_score_)

max_features = gs.best_params_['max_features']
max_features

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC 到这一步其实就可以停下了，因为复杂度和泛化误差的关系已经告诉我们，模型不能再进步了。

# COMMAND ----------

# MAGIC %md
# MAGIC #### min_samples_leaf

# COMMAND ----------

# MAGIC %md
# MAGIC #### min_samples_split

# COMMAND ----------

# MAGIC %md
# MAGIC #### criterion

# COMMAND ----------

param_grid = {'criterion':['gini', 'entropy']}

rfc = ensemble.RandomForestClassifier(n_estimators=n_estimators,random_state=random_state, max_depth=max_depth, max_features=max_features)
gs = GridSearchCV(rfc,param_grid,cv=10)
gs.fit(cancer.data,cancer.target)

# best_params_ 最佳参数组合
# best_score_ 最佳参数组合下的最佳准确率
print(gs.best_params_, gs.best_score_)

criterion = gs.best_params_['criterion']
criterion

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC 乳腺癌数据集非常完美，所以只需要调n_estimators一个参数就达到了随机森林在这个数据集上表现得极限。
