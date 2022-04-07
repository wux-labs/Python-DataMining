# Databricks notebook source
# MAGIC %md
# MAGIC # 红酒数据集

# COMMAND ----------

# MAGIC %md
# MAGIC ## 导入依赖

# COMMAND ----------

from sklearn import tree
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
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

print(type(wine))

# 这是一个数据字典结构的数据
wine

# COMMAND ----------

wine_contact = pd.concat([pd.DataFrame(wine.data),pd.DataFrame(wine.target)], axis=1)
wine_contact
# display(spark.createDataFrame(wine_contact))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 数据划分
# MAGIC 
# MAGIC 将数据划分成训练集和测试集，观察一下划分情况。

# COMMAND ----------

x_train, x_test, y_train, y_test = train_test_split(wine.data, wine.target, test_size=0.3)

# COMMAND ----------

x_train, x_test, y_train, y_test

# COMMAND ----------

# MAGIC %md
# MAGIC ## 模型训练
# MAGIC 
# MAGIC 训练模型并验证。

# COMMAND ----------

criterion="entropy"

# COMMAND ----------

clf = tree.DecisionTreeClassifier(criterion=criterion)
clf = clf.fit(x_train, y_train)

# 返回预测的准确度 accuracy
score = clf.score(x_test, y_test)

score

# COMMAND ----------

wine.feature_names

# COMMAND ----------

clf.feature_importances_

# COMMAND ----------

feature_names = ['酒精','苹果酸','灰','灰的碱性','镁','总酚','黄烷类','非黄烷类酚类','花青素','颜色强度','色调','od280/od315稀释葡萄酒','脯氨酸']
class_names = ['白兰地','威士忌','伏特加']

# COMMAND ----------

[*zip(feature_names, clf.feature_importances_)]

# COMMAND ----------

# MAGIC %md
# MAGIC ## 数据可视化

# COMMAND ----------

plt.figure(figsize=(20,10))

# filled=True，控制决策树的节点是否填充颜色
# rounded=True，控制决策树的节点是否使用圆角
# feature_names=feature_names，指定特征的名称
# class_names=['白兰地','威士忌','伏特加']，指定标签的名称
tree.plot_tree(clf,filled=True,rounded=True,feature_names=feature_names,class_names=class_names)

# COMMAND ----------

dot_data = tree.export_graphviz(clf, feature_names=feature_names, class_names=class_names, filled=True, rounded=True)

graph = graphviz.Source(dot_data)

graph

# COMMAND ----------

graph.render('/tmp/decision_tree_01', format="png")

dbutils.fs.cp("file:/tmp/decision_tree_01.png", "dbfs:/mnt/databrickscontainer1/")

# COMMAND ----------

df = spark.read.format("binaryFile").load("/mnt/databrickscontainer1/decision_tree_01.png")
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 参数调优

# COMMAND ----------

# MAGIC %md
# MAGIC ### 随机性参数

# COMMAND ----------

for i in range(0,20):
    clf = tree.DecisionTreeClassifier(criterion=criterion)
    clf = clf.fit(x_train, y_train)

    # 返回预测的准确度 accuracy
    score = clf.score(x_test, y_test)

    print(i , score)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC score会在某个值附近波动，每次画出来的树不一样。它为什么会不稳定？
# MAGIC 
# MAGIC 无论决策树模型如何进化，他在分枝上的本质都还是追求某个不纯度相关的指标的优化。不纯度是基于节点来计算的，也就是说，决策树在建树时，是靠优化节点来追求一棵优化的树的。sklearn表示，既然一棵树不能保证最优，那就建更多的不同的树，然后从中选取最好的。在每次分枝时，不使用全部特征，而是随机选取一部分特征，从中选取不纯度相关指标最优的作为分枝用的节点。这样，每次生成的树也就不同了。

# COMMAND ----------

# MAGIC %md
# MAGIC #### random_state
# MAGIC 
# MAGIC 用来设置分枝中的随机模式的参数，控制估计器的随机性。每次分枝时，这些要素始终是随机排列的。
# MAGIC 
# MAGIC 在高维度时随机性会表现更明显，低维度的数据（比如鸢尾花数据集），随机性几乎不会显现。设置任意整数，会一直生长出同一棵树，让模型稳定下来。

# COMMAND ----------

random_state = 0
max_score = 0

for i in range(0,20):
    clf = tree.DecisionTreeClassifier(criterion=criterion
                                      ,random_state=i)
    clf = clf.fit(x_train, y_train)

    # 返回预测的准确度 accuracy
    score = clf.score(x_test, y_test)
    
    if score > max_score and score != 1:
        max_score = score
        random_state = i

    print(i , score)

max_score, random_state

# COMMAND ----------

clf = tree.DecisionTreeClassifier(criterion=criterion
                                  ,random_state=random_state)
clf = clf.fit(x_train, y_train)

# 返回预测的准确度 accuracy
score = clf.score(x_test, y_test)

score

# COMMAND ----------

# MAGIC %md
# MAGIC #### splitter
# MAGIC 
# MAGIC 用于在每个节点上选择分枝的策略。
# MAGIC 
# MAGIC 用来控制决策树中的随机选项的。
# MAGIC 
# MAGIC * best：选择最佳分枝
# MAGIC * random：选择最佳随机分枝，树会更深，对训练集的拟合将会降低

# COMMAND ----------

splitter='random'

# COMMAND ----------

random_state = 0
max_score = 0

for i in range(0,2000):
    clf = tree.DecisionTreeClassifier(criterion=criterion
                                      ,random_state=i
                                      ,splitter=splitter)
    clf = clf.fit(x_train, y_train)

    # 返回预测的准确度 accuracy
    score = clf.score(x_test, y_test)
    
    if score > max_score and score != 1:
        max_score = score
        random_state = i

    if score > 0.97:
        print(i , score)

max_score, random_state

# COMMAND ----------

clf = tree.DecisionTreeClassifier(criterion=criterion
                                  ,random_state=random_state
                                  ,splitter=splitter)
clf = clf.fit(x_train, y_train)

# 返回预测的准确度 accuracy
score = clf.score(x_test, y_test)

score

# COMMAND ----------

# MAGIC %md
# MAGIC ### 剪枝参数
# MAGIC 
# MAGIC 在不加限制的情况下，一棵决策树会生长到衡量不纯度的指标最优，或者没有更多的特征可用为止，这样的决策树往往会过拟合。
# MAGIC 
# MAGIC 为了让决策树有更好的泛化性，我们需要对决策树进行剪枝。
# MAGIC 
# MAGIC **剪枝策略对决策树的影响巨大，正确的剪枝策略是优化决策树算法的核心。**

# COMMAND ----------

# MAGIC %md
# MAGIC #### max_depth
# MAGIC 
# MAGIC **int, default=None**
# MAGIC 
# MAGIC 树的最大深度。如果为 None，则节点将展开，树辉持续生长，直到所有叶子都是纯的，或者直到所有叶子包含的样本少于 min_samples_split。

# COMMAND ----------

max_depth = 0
max_score = 0

for i in range(1,20):
    clf = tree.DecisionTreeClassifier(criterion=criterion
                                      ,random_state=random_state
                                      ,splitter=splitter
                                      ,max_depth=i)
    clf = clf.fit(x_train, y_train)

    # 返回预测的准确度 accuracy
    score = clf.score(x_test, y_test)
    
    print(i , score)
    
    if score > max_score and score != 1:
        max_score = score
        max_depth = i

max_score, max_depth

# COMMAND ----------

clf = tree.DecisionTreeClassifier(criterion=criterion
                                  ,random_state=random_state
                                  ,splitter=splitter
                                  ,max_depth=max_depth)
clf = clf.fit(x_train, y_train)

# 返回预测的准确度 accuracy
score = clf.score(x_test, y_test)

score

# COMMAND ----------

# MAGIC %md
# MAGIC #### min_samples_split
# MAGIC 
# MAGIC **int or float, default=2**
# MAGIC 
# MAGIC 分枝内部节点所需的最小样本数。
# MAGIC 
# MAGIC 如果一个节点包含的样本数小于min_samples_split，这个节点就不会分枝，它会成为一个叶子节点。
# MAGIC 
# MAGIC * 如果为 int，则将其视为最小数。min_samples_split
# MAGIC * 如果为 float，则为比例，min_samples_splitceil(min_samples_split * n_samples)是每次分枝的最小样本数。
# MAGIC 
# MAGIC > 在版本 0.18 中更改： 为分数添加了浮点值。

# COMMAND ----------

for i in range(2,400):
    clf = tree.DecisionTreeClassifier(criterion=criterion
                                      ,random_state=random_state
                                      ,splitter=splitter
                                      ,max_depth=max_depth
                                      ,min_samples_split=i)
    clf = clf.fit(x_train, y_train)

    # 返回预测的准确度 accuracy
    score = clf.score(x_test, y_test)
    
    print(i , score)

# COMMAND ----------

clf = tree.DecisionTreeClassifier(criterion=criterion
                                  ,random_state=random_state
                                  ,splitter=splitter
                                  ,max_depth=max_depth
                                  ,min_samples_split=10)
clf = clf.fit(x_train, y_train)

# 返回预测的准确度 accuracy
score = clf.score(x_test, y_test)

score

# COMMAND ----------

# MAGIC %md
# MAGIC #### min_samples_leaf
# MAGIC 
# MAGIC **int or float, default=1**
# MAGIC 
# MAGIC 叶节点上所需的最小样本数。任何深度的分割点只有在每个左右分枝中至少留下min_samples_leaf个训练样本时，才会考虑该分割点。这可能会产生平滑模型的效果，尤其是在回归中。
# MAGIC 
# MAGIC * 如果为 int，则将其视为最小数。
# MAGIC * 如果为 float，则为比例，min_samples_leafceil(min_samples_leaf * n_samples)是每个节点的最小样本数。
# MAGIC 
# MAGIC > 在版本 0.18 中更改： 为分数添加了浮点值。

# COMMAND ----------

clf = tree.DecisionTreeClassifier(criterion=criterion
                                  ,random_state=random_state
                                  ,splitter=splitter
                                  ,max_depth=max_depth
                                  ,min_samples_split=9
                                  ,min_samples_leaf=4)
clf = clf.fit(x_train, y_train)

# 返回预测的准确度 accuracy
score = clf.score(x_test, y_test)

score
