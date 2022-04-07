# Databricks notebook source
# MAGIC %md
# MAGIC # 在合成数据集上的表现

# COMMAND ----------

# MAGIC %md
# MAGIC ## 导入依赖

# COMMAND ----------

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.tree import DecisionTreeClassifier

# COMMAND ----------

# MAGIC %md
# MAGIC ## 生成数据集

# COMMAND ----------

# make_classification 库生成随机的二分型数据
X, y = make_classification(n_samples=100,        # 生成100个样本
                          n_features=2,          # 包含2个特征
                          n_redundant=0,         # 添加冗余特征0个
                          n_informative=2,       # 包含信息的特征是2个
                          random_state=1,        # 随机模式1
                          n_clusters_per_class=1 # 每个簇内包含的标签类别有1个
                          )

# COMMAND ----------

# MAGIC %md
# MAGIC 观察一下数据

# COMMAND ----------

X,y

# COMMAND ----------

plt.scatter(X[:,0],X[:,1])

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC 从图上可以看出，生成的二分型数据的两个簇离彼此很远，这样不利于我们测试分类器的效果，因此我们使用np生成随机数，通过让已经生成的二分型数据加上一个 0~1 之间的随机数，使数据分布变得更分散。

# COMMAND ----------

rng = np.random.RandomState() # 生成一种随机模式
X += 2 * rng.uniform(size=X.shape) # 加减一个 0~1 的随机数

# COMMAND ----------

plt.scatter(X[:,0],X[:,1])

# COMMAND ----------

lineary_separable = (X, y)
datasets = [make_moons(noise=0.3, random_state=0),
            make_circles(noise=0.2, factor=0.5, random_state=1),
            lineary_separable]

# COMMAND ----------

# MAGIC %md
# MAGIC ## 观察数据集

# COMMAND ----------

# MAGIC %md
# MAGIC ### 生成三个数据集的图像

# COMMAND ----------

figure = plt.figure(figsize=(6, 9))

i = 1

for ds_index, ds in enumerate(datasets):
    X, y = ds
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=5)
    
    x0_min, x0_max = X[:,0].min() - 0.5, X[:,0].max() + 0.5
    x1_min, x1_max = X[:,1].min() - 0.5, X[:,1].max() + 0.5
    
    array1, array2 = np.meshgrid(np.arange(x0_min, x0_max, 0.2),
                                 np.arange(x1_min, x1_max, 0.2))
    
    # 生成彩色画布
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    
    # 在换不上加上一个圈子
    ax = plt.subplot(len(datasets), 2, i)
    
    if ds_index == 0:
        ax.set_title("输入数据")
    
    # 将数据集的分布放到坐标系上
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors='k')
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, edgecolors='k', alpha=0.6)
    
    # 设置坐标轴的最大值和最小值
    ax.set_xlim(array1.min(), array1.max())
    ax.set_ylim(array2.min(), array2.max())
    # 设定没有坐标轴
    ax.set_xticks(())
    ax.set_yticks(())
    
    i += 1
    
    ax = plt.subplot(len(datasets), 2, i)
    
    # 决策树的建模过程：实例化->fit训练->score接口得到预测的准确率
    clf = DecisionTreeClassifier(max_depth=5)
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    
    # 绘制决策边界，为此，我们将为网格中的每个点指定一种颜色[x0_min, x0_max] x [x1_min, x1_max]
    # 分类树的接口，predict_proba，返回每一个输入的数据点所对应的标签类概率
    # 类概率使数据点所在的叶节点中相同类的样本数量/叶节点中的样本总数量
    # 由于决策树在训练的时候导入的训练集X_train里面包含两个特征，所以我们在计算类概率的时候，也必须导入结构相同的数组，即是说，必须有两个特征
    # ravel()能够将一个多为数组转换成一维数组
    # np.c_ 是能够将两个数组组合起来的函数
    # 在这里，我们先将两个网格数据降维成一维数组，再将两个数组链接变成含有两个特征的数据
    Z = clf.predict_proba(np.c_[array1.ravel(), array2.ravel()])[:, 1]
    
    # 将返回的类概率作为数据，放到contourf里面绘制轮廓
    Z = Z.reshape(array1.shape)
    ax.contourf(array1, array2, Z, cmap=cm, alpha=0.8)

    if ds_index == 0:
        ax.set_title("决策树")
    
    # 将数据集的分布放到我们的坐标系上
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors='k')
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, edgecolors='k', alpha=0.6)

    # 设置坐标轴的最大值和最小值
    ax.set_xlim(array1.min(), array1.max())
    ax.set_ylim(array2.min(), array2.max())
    # 设定没有坐标轴
    ax.set_xticks(())
    ax.set_yticks(())
    
    # 写在右下角的数字
    ax.text(array1.max() - 0.3, array2.min() + 0.3, ('{:.1f}%'.format(score * 100)), size=15, horizontalalignment='right')
    
    i += 1

plt.tight_layout()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC 很容易看出，分类树天生不擅长环形数据，每个模型都有自己的决策上限，所以一个怎样调整都无法提升表现的可能性也是有的。
# MAGIC 
# MAGIC 当一个模型怎么调整都不行的时候，我们可以选择其他的模型使用。
# MAGIC 
# MAGIC * 最擅长月亮型数据的是最近邻算法、RBF支持向量机和高斯过程
# MAGIC * 最擅长环形数据的是最近邻算法和高斯过程
# MAGIC * 最擅长对半分的数据的是朴素贝叶斯、神经网络和随机森林
