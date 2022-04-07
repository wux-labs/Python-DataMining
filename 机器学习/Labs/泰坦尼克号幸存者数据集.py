# Databricks notebook source
# MAGIC %md
# MAGIC # 概述
# MAGIC 
# MAGIC 1912年4月15日，泰坦尼克号在首次航行期间撞上冰山后沉没，2224名乘客和机组人员中有1502人遇难。沉船导致大量伤亡的原因之一是没有足够的救生艇给乘客和船员。虽然幸存下来有一些运气因素，但有一些人比其他人更有可能生存，比如妇女，儿童和上层阶级。
# MAGIC 
# MAGIC **什么样的人在泰坦尼克号中更容易存活？**

# COMMAND ----------

# MAGIC %md
# MAGIC # 读取数据
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

# MAGIC %sh
# MAGIC 
# MAGIC ls -al /Workspace/Repos/wux_labs@outlook.com

# COMMAND ----------

import pandas as pd

df_train = pd.read_csv("../../Datasets/taitanic_train.csv")

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
# MAGIC # 基本预处理

# COMMAND ----------

# MAGIC %md
# MAGIC ## 缺失值处理
# MAGIC 
# MAGIC * **删除属性**或者**删除样本**：如果大部分样本该属性都缺失，这个属性能提供的信息有限，可以选择放弃使用该属性。
# MAGIC * **统计填充**：对于缺失值的属性，尤其是数值类型的属性，根据所有样本关于这维属性的统计值对其进行填充，如使用平均数、中位数、众数、最大值、最小值等，具体选择哪种统计值需要具体问题具体分析。
# MAGIC * **统一填充**：常用的统一填充值有：空、0、正无穷、负无穷等。
# MAGIC * **预测/模型填充**：可以通过预测模型利用不存在缺失值的属性来预测缺失值，如：K-NN。也就是先用预测模型把数据填充后再做进一步的工作，比如统计、学习等。虽然这种方法比较复杂，但是最后得到的结果比较好。
# MAGIC 
# MAGIC Pandas库：fillna()  
# MAGIC sklearn库：Imputer

# COMMAND ----------

# MAGIC %md
# MAGIC ### 几种填充场景
# MAGIC 
# MAGIC * 年收入：商品推荐场景下填充平均值
# MAGIC * 行为时间点：填充众数
# MAGIC * 价格：商品匹配场景下填充平均值
# MAGIC * 人体寿命：人口估计场景下填充平均值
# MAGIC * 驾龄：没有填写这一项的用户可能没有车，可以填充0
# MAGIC * 本科毕业时间：没有填写这一项的用户可能没有上大学，可以填充正无穷
# MAGIC * 婚姻状态：没有填写这一项的用户可能对自己的隐私比较敏感，可以单独设一个分类，如：1-已婚，0-未婚，9-未知

# COMMAND ----------

fill_age = df_train["Age"].fillna(value=df_train["Age"].mean())

# COMMAND ----------

# 旧版本的sklearn，用Imputer
# from sklearn.preprocessing import Imputer

# 新版本的sklearn，用SimpleImputer
from sklearn.impute import SimpleImputer

imp = SimpleImputer()

imp.fit(df_train[["Age"]])

fill_age = imp.transform(df_train[["Age"]])

fill_age

# COMMAND ----------

df_train.loc[:,"fill_age"] = fill_age

display(df_train.head())

df_train.info()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 数据无量纲化
# MAGIC 
# MAGIC 在机器学习算法实践中，我们往往有着将不同规格的数据转换到同一规格，或不将同分布的数据转换到某个特定分布的需求，这种需求统称为将数据“**无量纲化**”。
# MAGIC 
# MAGIC * 在梯度和矩阵为核心的算法中，譬如逻辑回归，支持向量机，神经网络，无量纲化可以加快求解速度
# MAGIC * 在距离类模型，譬如K近邻，K-Means聚类中，无量纲化可以帮我们提升模型精度，避免某一个取值范围特别大的特征对距离计算造成影响
# MAGIC 
# MAGIC 数据的无量纲化可以是线性的，也可以是非线性的。线性的无量纲化包括**中心化**处理和**缩放**处理（Scale）。
# MAGIC * 中心化的本质是让所有记录减去一个固定值，即让数据样本数据平移到某个位置。
# MAGIC * 缩放的本质是通过除以一个固定值，将数据固定在某个范围之中，取对数也算是一种缩放处理。

# COMMAND ----------

# MAGIC %md
# MAGIC ### 数值型特征
# MAGIC 
# MAGIC 数值型的幅度变换：对数变换、多项式变换。
# MAGIC * apply + numpy
# MAGIC * preprocessing scaler
# MAGIC 
# MAGIC 对数变换，使得特征和目标变量之间的关系更接近线性，从而提高预测效果。

# COMMAND ----------

import numpy as np

# apply函数主要是针对的列数和行数
log_age = df_train["fill_age"].apply(lambda x: np.log(x))

log_age

# COMMAND ----------

df_train.loc[:,"log_age"] = log_age

display(df_train.head())

df_train.info()

# COMMAND ----------

# MAGIC %md
# MAGIC #### 归一化
# MAGIC 
# MAGIC **归一化(Normalization)：** 当数据(x)按照最小值中心化后，再按极差（最大值 - 最小值）缩放，数据移动了最小值个单位，并且会被收敛到[0,1]之间，而这个过程，就叫做数据归一化
# MAGIC * 把数据变成[0,1]或者[-1,1]之间的小数。主要是为了数据处理方便提出来的，把数据映射到0~1范围之内处理，更加便捷快速。
# MAGIC * 把有量纲表达式变成无量纲表达式，便于不同单位或量级的指标能够进行比较和加权。
# MAGIC 
# MAGIC 归一化是一种简化计算的方式，即将有量纲的表达式，经过变换，化为无量纲的表达式，成为纯量。

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 最大最小归一化
# MAGIC 
# MAGIC **最大最小归一化(MinMaxScaler)：**
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=x^*%3D%5Cfrac%7Bx_i-x_%7Bmin%7D%7D%7Bx_%7Bmax%7D-x_%7Bmin%7D%7D)
# MAGIC 
# MAGIC 其中，![](https://www.zhihu.com/equation?tex=x_%7Bmin%7D)是样本中最小值，![](https://www.zhihu.com/equation?tex=x_%7Bmax%7D)是样本中最大值。
# MAGIC 
# MAGIC **注意**，在数据流场景下最大值与最小值是变化的。
# MAGIC 
# MAGIC **另外**，最大值与最小值非常容易受到异常点的影响，所以这种方法**鲁棒性**较差，只适合传统精确小数据场景。
# MAGIC 
# MAGIC > **鲁棒性**指控制系统在一定（结构，大小）的参数摄动下，维持其它某些性能的特性。  
# MAGIC 也就是**健壮性**。

# COMMAND ----------

from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

mm_scaler = MinMaxScaler()

fare_mm_trans = mm_scaler.fit_transform(df_train[['Fare']])

plt.figure(figsize=(20,40))
plt.scatter(df_train['Fare'].index,df_train['Fare'].values,c='b')
plt.scatter(df_train['Fare'].index,fare_mm_trans,c='r')

# COMMAND ----------

# MAGIC %md
# MAGIC #### 标准化
# MAGIC 
# MAGIC 在机器学习中，我们可能要处理不同种类的资料，例如，音讯和图片上的像素值，这些资料可能是高维度的，资料标准化后会使每个特征中的数值平均变为0(将每个特征的值都减掉原始资料中该特征的平均)、标准差变为1，这个方法被广泛的使用在许多机器学习算法中(例如：支持向量机、逻辑回归和类神经网络)。
# MAGIC 
# MAGIC **标准化(Standardization)：** 当数据(x)按均值(μ)中心化后，再按标准差(σ)缩放，数据就会服从为均值为0，方差为1的正态分布（即标准正态分布），而这个过程，就叫做**数据标准化**。
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=x^*%3D%5Cfrac%7Bx_i-%5Cmu%7D%7B%5Csigma%7D)
# MAGIC 
# MAGIC 原始特征值、均值、标准差，可以通过现有样本进行估计。在已有样本足够多的情况下比较稳定，适合现代嘈杂大数据场景。

# COMMAND ----------

from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

std_scaler = StandardScaler()

fare_std_trans = std_scaler.fit_transform(df_train[['Fare']])

plt.figure(figsize=(20,40))
plt.scatter(df_train['Fare'].index,df_train['Fare'].values,c='b')
plt.scatter(df_train['Fare'].index,fare_std_trans,c='r')

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC * 数据量小的话，不要求模型精确，可以使用最大最小归一化。
# MAGIC * 数据量大的话，要求模型精确，可以用标准化。
# MAGIC 
# MAGIC 大多数机器学习算法中，会选择StandardScaler来进行特征缩放，因为MinMaxScaler对异常值非常敏感。在PCA，聚类，逻辑回归，支持向量机，神经网络这些算法中，StandardScaler往往是最好的选择。
# MAGIC 
# MAGIC MinMaxScaler在不涉及距离度量、梯度、协方差计算以及数据需要被压缩到特定区间时使用广泛，比如数字图像处理中量化像素强度时，都会使用MinMaxScaler将数据压缩于[0,1]区间之中。

# COMMAND ----------

# MAGIC %md
# MAGIC #### 统计值
# MAGIC 
# MAGIC * max
# MAGIC * min
# MAGIC * quantile：分位数，是指在统计学中把所有数值由小到大排列并分成四等份，处于三个分割点位置的数值。

# COMMAND ----------

max_age = df_train['Age'].max()
min_age = df_train['Age'].min()

age_quantile_1 = df_train['Age'].quantile(0.25)
age_quantile_3 = df_train['Age'].quantile(0.75)

print("max_age:", max_age, " min_age:",min_age)
print("age_quantile_1:", age_quantile_1, " age_quantile_3:",age_quantile_3)


# COMMAND ----------

# MAGIC %md
# MAGIC #### 四则运算（本质上就是交叉特征/组合特征）
# MAGIC 
# MAGIC * 需要明白每个组合特征其实代表着A和B各自信息协同作用
# MAGIC * 一个更好地诠释好的交叉特征的实例是类似于（经度，纬度），一个相同的经度对应了地图上的很多地方，维度也一样，但是一旦你将经度和纬度组合到一起，它们就代表了地理上特定的区域，区域中的每一部分是拥有着类似的特性

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC SibSp：堂兄弟的个数 + 父母孩子的个数 + 自己 = family 家庭组合特征

# COMMAND ----------

display(df_train.head())

df_train.loc[:,'family_size'] = df_train['SibSp'] + df_train['Parch'] + 1 # 1 代表自己

display(df_train.head())

# COMMAND ----------

# MAGIC %md
# MAGIC #### 高次特征与交叉特征
# MAGIC 
# MAGIC PolynomialFeatures
# MAGIC 
# MAGIC * degree：控制多项式的次数
# MAGIC * nteraction_only：默认为 False，如果指定为 True，那么久不会有特征自己和自己结合的项
# MAGIC * include_bias：默认为 True，如果为 True，那么结果中就会有 0 次幂项，即全为 1 这一列

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 多项式特征变换
# MAGIC 
# MAGIC 目标是将特征两两组合起来，使得特征和目标之间的关系更接近线性，从而提高预测的效果。

# COMMAND ----------

from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2)

print("原始特征")
display(df_train[['SibSp','Parch']])

poly_fea = poly.fit_transform(df_train[['SibSp','Parch']])

poly_fea

# COMMAND ----------

# MAGIC %md
# MAGIC #### 离散化
# MAGIC 
# MAGIC pandas cut
# MAGIC 
# MAGIC 将数值类型属性转换成类别，能使算法减少噪声的干扰，通过将一定范围内的数值划分成确定的块。
# MAGIC 
# MAGIC 举个例子，预测一个人是否拥有某款衣服，这里年龄是一个确切的因子。其实年龄组是更为相关的因子，所以可以将年龄划分成1-10，11-18，19-25，26-40等等，因为相近的年龄组表现出相似的属性。

# COMMAND ----------

df_train.loc[:,'fare_cut'] = pd.cut(df_train['Fare'],5) # 分成4个区间

df_train.head()

# COMMAND ----------

df_train['fare_cut'].unique()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 类别型特征
# MAGIC 
# MAGIC * 类别型：OneHot encoding/独热向量编码
# MAGIC   * pandas get_dummies/哑变量
# MAGIC   * OneHotEncoder()
# MAGIC 
# MAGIC * 独热编码：使用一个二进制的位来表示某个定性特征的出现与否
# MAGIC 
# MAGIC 一些属性是类别型而不是离散型，举一个简单的例子，由（红、绿、蓝）组成的颜色属性，最常用的方式是把每个类别属性转换成二元属性，即从{0, 1}取一个值。因此基本上增加的属性等于相应数目的类别，并且对于数据集中的每个实例，只有一个1（其他的为0），这也就是独热（one-hot）编码方式。

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC * 对于数值型的特征中，特征的属性是无序的，用独热编码/哑变量，比如：性别、颜色、星期
# MAGIC * 对于数值型的特征中，特征的属性是有序的，用标签编码（LabelEncoder），比如：公司的成立时间
