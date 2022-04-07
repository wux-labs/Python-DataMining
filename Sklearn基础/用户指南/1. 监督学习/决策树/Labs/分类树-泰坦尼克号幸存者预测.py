# Databricks notebook source
# MAGIC %md
# MAGIC # 泰坦尼克号幸存者预测
# MAGIC 
# MAGIC 1912年4月15日，泰坦尼克号在首次航行期间撞上冰山后沉没，2224名乘客和机组人员中有1502人遇难。沉船导致大量伤亡的原因之一是没有足够的救生艇给乘客和船员。虽然幸存下来有一些运气因素，但有一些人比其他人更有可能生存，比如妇女，儿童和上层阶级。
# MAGIC 
# MAGIC **什么样的人在泰坦尼克号中更容易存活？**

# COMMAND ----------

# MAGIC %md
# MAGIC ## 导入依赖

# COMMAND ----------

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV

# COMMAND ----------

# MAGIC %md
# MAGIC ## 读取数据
# MAGIC 
# MAGIC 拿到一批数据，先用3个函数查看一下
# MAGIC 
# MAGIC * head() 查看行数，默认是5行
# MAGIC * info() 查看行数以及数据类型
# MAGIC * describe() 查看每列的均值、方差、分位数等
# MAGIC 
# MAGIC 其中：
# MAGIC 
# MAGIC * PassengerId：乘客的ID号。
# MAGIC * Survived：生存的标志。
# MAGIC * Pclass：船舱等级。
# MAGIC * Name：名字。
# MAGIC * Sex：性别，女士优先。
# MAGIC * Age：年龄，因为优先保护老幼。
# MAGIC * SibSp：兄弟姐妹，就是有些人和兄弟姐妹一起上船的。有可能因为救他们而导致自己没有上救生船船。
# MAGIC * Parch：父母和小孩，就是有些人会带着父母小孩上船的。有可能因为要救父母小孩耽误上救生船。
# MAGIC * Ticket：票的编号。
# MAGIC * Fare：费用。
# MAGIC * Cabin：舱号。
# MAGIC * Embarked：上船的地方。

# COMMAND ----------

df_train = pd.read_csv("../../../../../Datasets/taitanic_train.csv")

# COMMAND ----------

df_train.shape

# COMMAND ----------

# head() 查看行数，默认是5行
df_train.head()

# COMMAND ----------

# info() 查看行数以及数据类型
df_train.info()

# COMMAND ----------

# describe() 查看每列的均值、方差、分位数等
df_train.describe()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 基本预处理

# COMMAND ----------

# MAGIC %md
# MAGIC ### 筛选特征
# MAGIC 
# MAGIC 这里可以将对本次探索没有用的特征删除，比如"Name","Ticket","Cabin"，对于缺失值比较多的特征也可以删除。

# COMMAND ----------

# inplace=True，指定用新的结果替换旧的数据
# df_train = df_train.drop(["Name","Ticket","Cabin"], inplace=False, axis=1)
df_train.drop(["Name","Ticket","Cabin"], inplace=True, axis=1)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 缺失值处理

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Age的缺失值比较多，需要进行填充。

# COMMAND ----------

# 使用平均值填充年龄的缺失值
df_train["Age"] = df_train["Age"].fillna(value=df_train["Age"].mean())

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Embarked的缺失值是2条，可以直接删除这两条记录。
# MAGIC 
# MAGIC 对于缺失值比较少的情况，可以直接删除有缺失值的记录。

# COMMAND ----------

df_train = df_train.dropna()

# COMMAND ----------

# info() 查看行数以及数据类型
df_train.info()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 分类数据处理
# MAGIC 
# MAGIC 决策树只能处理数值类型，需要将数据集中的字符数据都转换成数值类型的数据。

# COMMAND ----------

df_train["Embarked"].unique()

# COMMAND ----------

# 获取Embarked的值并转换成列表
labels = df_train["Embarked"].unique().tolist()
# 将Embarked的值转换成对应的列表中的索引
df_train["Embarked"] = df_train["Embarked"].apply(lambda x: labels.index(x))

# COMMAND ----------

df_train["Sex"].unique()

# COMMAND ----------

df_train["Sex"] = (df_train["Sex"] == "male").astype("int")

# COMMAND ----------

# head() 查看行数，默认是5行
df_train.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 提取特征和标签
# MAGIC 
# MAGIC 由于决策树需要将特征和标签分别进行传入，所以需要将数据集中的特征和标签分别提取出来。

# COMMAND ----------

X = df_train.iloc[:, df_train.columns != "Survived"]
y = df_train.iloc[:, df_train.columns == "Survived"]

X, y

# COMMAND ----------

# MAGIC %md
# MAGIC ## 数据划分
# MAGIC 
# MAGIC 将数据划分成训练集和测试集，观察一下划分情况。

# COMMAND ----------

# 对数据集进行拆分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

X_train, X_test, y_train, y_test

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC 可以发现上面的数据集拆分后的训练集和测试集的索引发生了变化，这对我们以后按索引进行取数、查看等会造成困扰，所以需要对数据进行重新索引。

# COMMAND ----------

# 对数据集的索引重新索引
for i in [X_train, X_test, y_train, y_test]:
    i.index = range(i.shape[0])

X_train, X_test, y_train, y_test

# COMMAND ----------

# MAGIC %md
# MAGIC ## 模型训练

# COMMAND ----------

clf = DecisionTreeClassifier(criterion="entropy", random_state=25)
clf = clf.fit(X_train, y_train)

# 返回预测的准确度 accuracy
score = clf.score(X_test, y_test)

score

# COMMAND ----------

clf = DecisionTreeClassifier(criterion="entropy", random_state=25)
score = cross_val_score(clf, X, y, cv=10).max()
score

# COMMAND ----------

# MAGIC %md
# MAGIC 从上面的结果看出，模型训练的结果并不理想。

# COMMAND ----------

# MAGIC %md
# MAGIC ## 参数调优

# COMMAND ----------

train = []
test = []

for i in range(100):
    clf = DecisionTreeClassifier(random_state=i)
    clf = clf.fit(X_train, y_train)
    train.append(clf.score(X_train, y_train))
    test.append(cross_val_score(clf, X, y, cv=10).mean())

plt.plot(range(1,101), train, color="red", label="训练集")
plt.plot(range(1,101), test, color="blue", label="测试集")

plt.legend()

# COMMAND ----------

# MAGIC %md
# MAGIC 从上图看，random_state怎么调整对结果都没太大的影响

# COMMAND ----------

train = []
test = []

for i in range(30):
    clf = DecisionTreeClassifier(criterion="entropy", random_state=5, max_depth=i+1)
    clf = clf.fit(X_train, y_train)
    train.append(clf.score(X_train, y_train))
    test.append(cross_val_score(clf, X, y, cv=10).mean())

plt.plot(range(1,31), train, color="red", label="训练集")
plt.plot(range(1,31), test, color="blue", label="测试集")

plt.legend()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC 从上图看，max_depth越大过拟合的情况越严重，max_depth>25以后达到最严重的过拟合，max_depth=3的时候测试集和训练集上的分数最接近并且测试集上的分数最高。

# COMMAND ----------

# MAGIC %md
# MAGIC ### 网格搜索
# MAGIC 
# MAGIC 能够帮助我们同时调整多个参数的技术，枚举技术。

# COMMAND ----------

gini_threholds = np.linspace(0,0.5,20)
# entropy_threholds = np.linspace(0,1,20)

parameters = {"criterion":("gini","entropy")
             ,"splitter":("best","random")
             ,"max_depth":[*range(1,10)]
             ,"min_samples_leaf":[*range(1,50,5)]
#              ,"min_impurity_decrease":[*gini_threholds]
             }

clf = DecisionTreeClassifier(random_state=5)
gs = GridSearchCV(clf, parameters, cv=10)
gs.fit(X_train, y_train)

# COMMAND ----------

# 从我们输入的参数和参数取值的列表中，返回最佳组合
gs.best_params_

# COMMAND ----------

# 网格搜索后的模型的评判标准
gs.best_score_

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC 从上面的过程可以看出，该数据集使用决策树算法可能不是很好。
