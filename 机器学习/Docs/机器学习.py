# Databricks notebook source
# MAGIC %md
# MAGIC # 读取数据
# MAGIC 
# MAGIC 拿到一批数据，先用3个函数查看一下
# MAGIC * head() 查看行数，默认是5行
# MAGIC * info() 查看行数以及数据类型
# MAGIC * describe() 查看每列的均值、方差、分位数等

# COMMAND ----------

# MAGIC %md
# MAGIC # 数据预处理
# MAGIC 
# MAGIC 数据预处理是从数据中检测，纠正或删除损坏，不准确或不适用于模型的记录的过程。
# MAGIC 
# MAGIC 可能面对的问题有：
# MAGIC * 数据类型不同，比如有的是文字，有的是数字，有的含时间序列，有的连续，有的间断。
# MAGIC * 数据的质量不行，有噪声，有异常，有缺失，数据出错，量纲不一，有重复，数据是偏态，数据量太大或太小
# MAGIC 
# MAGIC 数据预处理的目的：让数据适应模型，匹配模型的需求

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
# MAGIC ### 归一化
# MAGIC 
# MAGIC **归一化(Normalization)：** 当数据(x)按照最小值中心化后，再按极差（最大值 - 最小值）缩放，数据移动了最小值个单位，并且会被收敛到[0,1]之间，而这个过程，就叫做数据归一化
# MAGIC * 把数据变成[0,1]或者[-1,1]之间的小数。主要是为了数据处理方便提出来的，把数据映射到0~1范围之内处理，更加便捷快速。
# MAGIC * 把有量纲表达式变成无量纲表达式，便于不同单位或量级的指标能够进行比较和加权。
# MAGIC 
# MAGIC 归一化是一种简化计算的方式，即将有量纲的表达式，经过变换，化为无量纲的表达式，成为纯量。

# COMMAND ----------

# MAGIC %md
# MAGIC #### 最大最小归一化
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

data = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]

# COMMAND ----------

import numpy as np

# COMMAND ----------

X = np.array(data)

# COMMAND ----------

# 归一化
X_nor = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
X_nor

# COMMAND ----------

#逆转归一化
X_returned = X_nor * (X.max(axis=0) - X.min(axis=0)) + X.min(axis=0)
X_returned

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC 在sklearn当中，使用preprocessing.MinMaxScaler来实现这个功能。MinMaxScaler有一个重要参数，feature_range，控制我们希望把数据压缩到的范围，默认是[0,1]。

# COMMAND ----------

from sklearn.preprocessing import MinMaxScaler

# COMMAND ----------

# 实现归一化

# 实例化
scaler = MinMaxScaler()
# fit，在这里本质是生成min(x)和max(x)
scaler = scaler.fit(data)
# 通过接口导出结果
result = scaler.transform(data)
result

# COMMAND ----------

# 训练和导出结果一步达成
result = scaler.fit_transform(data)
# 将归一化后的结果逆转
result = scaler.inverse_transform(result)

# 当X中的特征数量非常多的时候，fit会报错并表示，数据量太大了我计算不了
# 此时使用partial_fit作为训练接口
# scaler = scaler.partial_fit(data)

# COMMAND ----------

# 使用MinMaxScaler的参数feature_range实现将数据归一化到[0,1]以外的范围中
# 实例化
scaler = MinMaxScaler(feature_range=[5,10])
# 训练和导出结果一步达成
result = scaler.fit_transform(data)
result

# COMMAND ----------

# MAGIC %md
# MAGIC ### 标准化
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

# COMMAND ----------

data = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]

# COMMAND ----------

# 实例化
scaler = StandardScaler()
# fit，本质是生成均值和方差
scaler.fit(data)

# mean_，查看均值的属性
# var_，查看方差的属性
scaler.mean_, scaler.var_

# COMMAND ----------

# 通过接口导出结果
x_std = scaler.transform(data)

# 导出的结果是一个数组
# 用mean()查看均值
# 用std()查看方差
x_std.mean(), x_std.std()

# COMMAND ----------

# 使用fit_transform(data)一步达成结果
scaler.fit_transform(data)

# COMMAND ----------

# 使用inverse_transform逆转标准化
scaler.inverse_transform(x_std)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC 对于StandardScaler和MinMaxScaler来说，空值NaN会被当做是缺失值，在fit的时候忽略，在transform的时候保持缺失NaN的状态显示。并且，尽管去量纲化过程不是具体的算法，但在fit接口中，依然只允许导入至少二维数组，一维数组导入会报错。通常来说，我们输入的X会是我们的特征矩阵，现实案例中特征矩阵不太可能是一维所以不会存在这个问题。

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC * 数据量小的话，不要求模型精确，可以使用最大最小归一化。
# MAGIC * 数据量大的话，要求模型精确，可以用标准化。
# MAGIC 
# MAGIC 大多数机器学习算法中，会选择StandardScaler来进行特征缩放，因为MinMaxScaler对异常值非常敏感。在PCA，聚类，逻辑回归，支持向量机，神经网络这些算法中，StandardScaler往往是最好的选择。
# MAGIC 
# MAGIC MinMaxScaler在不涉及距离度量、梯度、协方差计算以及数据需要被压缩到特定区间时使用广泛，比如数字图像处理中量化像素强度时，都会使用MinMaxScaler将数据压缩于[0,1]区间之中。
# MAGIC 
# MAGIC | 无量纲化 | 功能 | 中心化 | 缩放 | 描述 |
# MAGIC | ----- | ----- | ----- | ----- | ----- |
# MAGIC | StandardScaler | 标准化 | 均值 | 方差 | 通过减掉均值并将数据缩放到单位方差来标准化特征，标准化完毕后的特征服从标准正太分布，即方差为1，均值为0 |
# MAGIC | MinMaxScaler | 归一化 | 最小值 | 极差 | 通过最大值与最小值将每个特征缩放到给定的范围，默认[0,1] |
# MAGIC | MaxAbsScaler | 缩放 | N/A | 最大值 | 通过让每一个特征里的数据，除以该特征中绝对值最大的数值的绝对值，将数据压缩到[-1,1]之间。这种做法并没有中心化数据，因此不会破坏数据的稀疏性。数据的稀疏性是指，数据中包含0的比例，0越多，数据越稀疏 |
# MAGIC | RobustScaler | 无量纲化 | 中位数 | 四分位数范围 | 使用可以处理异常值，对异常值不敏感的统计量来缩放数据。这个缩放器删除中位数并根据百分位数范围（IQR:Interquartile Range）缩放数据。IQR是第一分位数（25%）和第三分位数（75%）之间的范围。数据集的标准化是通过去除均值，缩放到单位方差来完成，但是异常值通常会对样本的均值和方差造成负面影响，当异常值很多噪音很大时，用中位数和四分位数范围通常会产生更好的效果 |
# MAGIC | Normalizer | 无量纲化 | N/A | sklearn中未明确，依范数原理应当是：<br>L1：样本向量的长度/样本中每个元素绝对值的和<br>L2：样本向量的长度/样本中每个元素的欧式距离 | 将样本独立缩放到单位范数。每个至少带一个非0值的样本都会被独立缩放，使得整个样本的长度都为L1范数或L2范数。这个类可以处理密集数组（numpy arrays）或scipy中的稀疏矩阵（scipy.sparse），如果你希望避免复制/转换过程中的负担，请使用CSR格式的矩阵。<br>将输入的数据缩放到单位范数是文本分类或聚类中的常见操作。例如，两个L2正则化后的TF-IDF向量的点积是向量的余弦相似度，并且是信息检索社区常用的向量空间模型的基本相似性度量。<br>使用参数norm来确定要正则化的范数方向，可以选择L1、L2以及max三种选项，默认L2范数。<br>这个评估器的fit接口什么也不做，但在管道中使用依然是很有用的 |
# MAGIC | PowerTransformer | 非线性无量纲化 | N/A | N/A | 应用特征功率变换使数据更接近正太分布。<br>功率变换是一系列参数单调变换，用于使数据更像高斯。对于建模与异方差（非常数方差）或其他需要正太性的情况相关的问题非常有用。要求输入的数据严格为正，power_transform()通过最大似然估计来稳定方差和并确定最小化偏度的最佳参数。<br>默认情况下，标准化应用于转换后的数据 |
# MAGIC | QuantileTransformer | 非线性无量纲化 | N/A | N/A | 使用百分位数转换特征，通过缩小边缘异常值和非异常值之间的距离来提供特征的非线性变换。可以使用参数output_distribution="normal"来将数据映射到标准正太分布 |
# MAGIC | KernelCenterer | 中心化 | 均值 | N/A | 将核矩阵中心化 |

# COMMAND ----------

# MAGIC %md
# MAGIC ## 特征数据处理

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

# MAGIC %md
# MAGIC #### 统计值
# MAGIC 
# MAGIC * max
# MAGIC * min
# MAGIC * quantile：分位数，是指在统计学中把所有数值由小到大排列并分成四等份，处于三个分割点位置的数值。
# MAGIC 
# MAGIC 场景：
# MAGIC 
# MAGIC * 借贷额度场景下计算最小值。
# MAGIC * 保险费用估计场景下计算最大值。

# COMMAND ----------

# MAGIC %md
# MAGIC #### 四则运算（本质上就是交叉特征/组合特征）
# MAGIC 
# MAGIC * 需要明白每个组合特征其实代表着A和B各自信息协同作用
# MAGIC * 一个更好地诠释好的交叉特征的实例是类似于（经度，纬度），一个相同的经度对应了地图上的很多地方，维度也一样，但是一旦你将经度和纬度组合到一起，它们就代表了地理上特定的区域，区域中的每一部分是拥有着类似的特性

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

# MAGIC %md
# MAGIC #### 离散化
# MAGIC 
# MAGIC pandas cut
# MAGIC 
# MAGIC 将数值类型属性转换成类别，能使算法减少噪声的干扰，通过将一定范围内的数值划分成确定的块。
# MAGIC 
# MAGIC 举个例子，预测一个人是否拥有某款衣服，这里年龄是一个确切的因子。其实年龄组是更为相关的因子，所以可以将年龄划分成1-10，11-18，19-25，26-40等等，因为相近的年龄组表现出相似的属性。

# COMMAND ----------

# MAGIC %md
# MAGIC ### 类别型特征
# MAGIC 
# MAGIC 在机器学习中，大多数算法，譬如逻辑回归，支持向量机SVM，K近邻算法等都只能够处理数值型数据，不能处理文字，在sklearn当中，除了专用来处理文字的算法，其他算法在fit的时候全部要求输入数组或矩阵，也不能够导入文字型数据（其实手写决策树和普斯贝叶斯可以处理文字，但是sklearn中规定必须导入数值型）。然而在现实中，许多标签和特征在数据收集完毕的时候，都不是以数字来表现的。
# MAGIC 
# MAGIC 在这种情况下，为了让数据适应算法和库，我们必须将数据进行编码，即是说，将文字型数据转换为数值型。

# COMMAND ----------

import pandas as pd

taitanic_train = pd.read_csv("../../Datasets/taitanic_train.csv")

# 缺失值全部删除记录
taitanic_train.dropna(inplace=True)
# 行记录被删除后，行index不会变，会导致index不连续，对后续基于index的处理会导致问题，所以需要将index重置一下
taitanic_train_reindex = taitanic_train.reset_index()

taitanic_train["SurvivedLabel"] = taitanic_train["Survived"].map(lambda x: "Yes" if x == 1 else "No")

print(taitanic_train.info(), taitanic_train.shape)

taitanic_train.head()

# COMMAND ----------

# MAGIC %md
# MAGIC #### 标签处理
# MAGIC 
# MAGIC preprocessing.LabelEncoder：标签专用，能够将分类转换为分类数值。

# COMMAND ----------

from sklearn.preprocessing import LabelEncoder

taitanic_train_label = taitanic_train.copy()

# 要输入的是标签
y = taitanic_train_label.iloc[:,-1]

# 实例化
le = LabelEncoder()
le = le.fit(y)
label = le.transform(y)

# label = le.fit_transform(y)
# label = LabelEncoder().fit_transform(taitanic_train_label.iloc[:,-1])
# classes_ 查看标签中究竟有多少类别
y, le.classes_, label

# COMMAND ----------

taitanic_train_label["SurvivedLabelEncoder"] = label

taitanic_train_label.head()

# COMMAND ----------

# MAGIC %md
# MAGIC #### 特征处理
# MAGIC 
# MAGIC preprocessing.OrdinalEncoder：特征专用，能够将分类特征转换为分类数值。

# COMMAND ----------

from sklearn.preprocessing import OrdinalEncoder

taitanic_train_feature = taitanic_train.copy()

# categories_ 查看标签中究竟有多少类别，[array(['female', 'male'], dtype=object)]
OrdinalEncoder().fit(taitanic_train_feature.iloc[:,4:5]).categories_

taitanic_train_feature.iloc[:,4:5], OrdinalEncoder().fit_transform(taitanic_train_feature.iloc[:,4:5])

# COMMAND ----------

# categories_ 查看标签中究竟有多少类别，[array(['C', 'Q', 'S'], dtype=object)]
OrdinalEncoder().fit(taitanic_train_feature.iloc[:,11:12]).categories_

taitanic_train_feature.iloc[:,11:12], OrdinalEncoder().fit_transform(taitanic_train_feature.iloc[:,11:12])

# COMMAND ----------

# MAGIC %md
# MAGIC #### 思考
# MAGIC 
# MAGIC 我们刚才已经用OrdinalEncoder把分类变量Sex和Embarked都转换成数字对应的类别了。在Embarked这一列中，我们使用[0,1,2]代表了三个不同的登船地，然而这种转换是正确的吗？
# MAGIC 
# MAGIC 我们来思考三种不同性质的分类数据：
# MAGIC 
# MAGIC 1） 登船地（S，C，Q）
# MAGIC 
# MAGIC 三种取值S，C，Q是相互独立的，彼此之间完全没有联系，表达的是S≠C≠Q的概念。这是**名义变量**。
# MAGIC 
# MAGIC 2） 学历（小学，初中，高中）
# MAGIC 
# MAGIC 三种取值不是完全独立的，我们可以明显看出，在性质上可以有高中>初中>小学这样的联系，学历有高低，但是学历取值之间却不是可以计算的，我们不能说小学 + 某个取值 = 初中。这是**有序变量**。
# MAGIC 
# MAGIC 3） 体重（>45kg，>90kg，>135kg）
# MAGIC 
# MAGIC 各个取值之间有联系，且是可以互相计算的，比如120kg - 45kg = 90kg，分类之间可以通过数学计算互相转换。这是**有距变量**。
# MAGIC 
# MAGIC 然而在对特征进行编码的时候，这三种分类数据都会被我们转换为[0,1,2]，这三个数字在算法看来，是连续且可以计算的，这三个数字相互不等，有大小，并且有着可以相加相乘的联系。所以算法会把登船地，学历这样的分类特征，都误会成是体重这样的分类特征。这是说，我们把分类转换成数字的时候，忽略了数字中自带的数学性质，所以给算法传达了一些不准确的信息，而这会影响我们的建模。
# MAGIC 
# MAGIC 类别OrdinalEncoder可以用来处理有序变量，但对于名义变量，我们只有使用**哑变量**的方式来处理，才能够尽量向算法传达最准确的信息：
# MAGIC 
# MAGIC | 原始分类 | 分类数值 | 哑变量 |
# MAGIC | ----- | ----- | ----- |
# MAGIC | C | 0 | [0, 0, 1] |
# MAGIC | Q | 1 | [0, 1, 0] |
# MAGIC | S | 2 | [1, 0, 0] |
# MAGIC 
# MAGIC 这样的变化，让算法能够彻底领悟，原来三个取值是没有可计算性质的，是“有你就没有我”的不等概念。在我们的数据中，Sex和Embarked，都是这样的名义变量。因此我们需要使用**独热编码**，将两个特征都转换为**哑变量**。

# COMMAND ----------

# MAGIC %md
# MAGIC #### 独热编码
# MAGIC 
# MAGIC 为了解决上述问题，其中一种可能的解决方法是采用**独热编码（One-Hot Encoding）**。独热编码即 One-Hot 编码，又称一位有效编码，其方法是使用N位状态寄存器来对N个状态进行编码，每个状态都由他独立的寄存器位，并且在任意时候，其中只有一位有效。
# MAGIC 
# MAGIC > 例如，对六个状态(0,1,2,3,4,5)进行编码：  
# MAGIC 自然顺序码为 000,001,010,011,100,101  
# MAGIC 独热编码则是 000001,000010,000100,001000,010000,100000  
# MAGIC 
# MAGIC 在回归，分类，聚类等机器学习算法中，特征之间距离的计算或相似度的计算是非常重要的。而常用的距离或相似度的计算都是在欧式空间的相似度计算，计算余弦相似性，基于的就是欧式空间。
# MAGIC 
# MAGIC 使用独热编码（One-Hot Encoding），将离散特征的取值扩展到了欧式空间，离散特征的某个取值就对应欧式空间的某个点。将离散型特征使用独热编码（One-Hot Encoding），会让特征之间的距离计算更加合理。
# MAGIC 
# MAGIC 独热编码优缺点：
# MAGIC * 优点：独热编码解决了分类器不好处理属性数据的问题，在一定程度上也起到了扩充特征的作用。它的值只有0和1，不同的类型存储在垂直的空间。
# MAGIC * 缺点：当类别的数量很多时，特征空间会变得非常大。在这种情况下，一般可以用PCA来减少维度。而且one hot encoding+PCA这种组合在实际中也非常有用。

# COMMAND ----------

# MAGIC %md
# MAGIC #### 哑变量
# MAGIC 
# MAGIC 通过独热编码，可以将一个具有N个分类属性的特征，扩充到N个具有0、1值的特征。
# MAGIC 
# MAGIC **哑变量（Dummy Variable）**，又称为**虚拟变量**、**虚设变量**或**名义变量**，从名称上看就知道，它是人为虚设的变量，通常取值为0或1，来反映某个变量的不同属性。对于有n个分类属性的自变量，通常需要选取1个分类作为参照，因此可以产生n-1个哑变量。
# MAGIC 
# MAGIC 将哑变量引入回归模型，虽然使模型变得较为复杂，但可以更直观地反映出该自变量的不同属性对于因变量的影响，提高了模型的精度和准确度。

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC * 对于数值型的特征中，特征的属性是无序的，用独热编码/哑变量，比如：性别、颜色、星期
# MAGIC * 对于数值型的特征中，特征的属性是有序的，用标签编码（LabelEncoder），比如：公司的成立时间

# COMMAND ----------

from sklearn.preprocessing import OneHotEncoder

taitanic_train_onehot = taitanic_train.copy()

taitanic_train_onehot.head()

# COMMAND ----------

X = taitanic_train_onehot.iloc[:,11:12]

enc = OneHotEncoder(categories='auto').fit(X)
result = enc.transform(X).toarray()
result.shape

# COMMAND ----------

# 也可以一步到位
OneHotEncoder(categories='auto').fit_transform(X).toarray()

# COMMAND ----------

# 获取新特征的名称
enc.get_feature_names()

# COMMAND ----------

# axis=1，表示跨行进行合并，也就是将量表左右相连，如果是axis=0，就是将量表上下相连
# pd.concat(axis=1)，基于行index值对数据集进行拼接，所以要求被拼接的数据集具有相同的行index，如果行index不同，则拼接后的结果具有空行数据
display(pd.concat([taitanic_train_onehot,pd.DataFrame(result)],axis=1))
display(pd.concat([taitanic_train_onehot.reset_index(),pd.DataFrame(result)],axis=1))

# COMMAND ----------

# MAGIC %md
# MAGIC 特征可以做哑变量，标签也可以吗？可以，使用类sklearn.preprocessing.LabelBinarizer可以对标签做哑变量，许多算法都可以处理多标签问题（比如说决策树），但是这样的做法在现实中不常见。

# COMMAND ----------

# MAGIC %md
# MAGIC #### 总结
# MAGIC 
# MAGIC | 编码与哑变量 | 功能 | 重要参数 | 重要属性 | 重要接口 |
# MAGIC | ----- | ----- | ----- | ----- | ----- |
# MAGIC | LabelEncoder| 分类标签编码 | N/A | classes_：查看标签中究竟有多少类别 | fit<br>transform<br>fit_transform<br>inverse_transform |
# MAGIC | OrdinalEncoder | 分类特征编码 | N/A | categories_：查看特征中究竟有多少类别 | fit<br>transform<br>fit_transform<br>inverse_transform |
# MAGIC | OneHotEncoder | 独热编码，为名义变量创建哑变量 | **categories**：每个特征都有哪些类别，默认"auto"表示让算法自己判断，或者可以输入列表，每个元素都是一个列表，表示每个特征中的不同类别 <br><br>**handle_unknown**：当输入了categories，且算法遇见了categories中没有写明的特征或类别时，是否报错。默认"error"表示报错，也可以选择"ignore"表示请忽视。如果选择"ignore"则未在categories中注明的特征或类别的哑变量会全部显示为0。在inverse_transform中，未知特征或类别会被返回None。 | categories_：查看特征中究竟有多少类别 | fit<br>transform<br>fit_transform<br>inverse_transform<br>get_feature_names：查看生成的哑变量的每一列都是什么特征的什么取值 |

# COMMAND ----------

# MAGIC %md
# MAGIC ### 连续型特征

# COMMAND ----------

# MAGIC %md
# MAGIC #### 二值化
# MAGIC 
# MAGIC 根据阈值将数据二值化（将特征值设置为0或1），用于处理连续型变量。大于阈值的值映射为1，而小于或等于阈值的值映射为0。默认阈值为0时，特征中所有的正值都映射到1。二值化是对文本计数数据的常见操作，分析人员可以决定仅考虑某种现象的存在与否。它还可以用作考虑布尔随机变量的估计器的预处理步骤（例如，使用贝叶斯设置中的伯努利分布建模）。

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Binarizer

# COMMAND ----------

from sklearn.preprocessing import Binarizer

taitanic_train_binary = taitanic_train.copy()

X = taitanic_train_binary.iloc[:,5].values.reshape(-1,1)
transformer = Binarizer(threshold=32).fit_transform(X)

data = pd.concat([pd.DataFrame(X), pd.DataFrame(transformer)],axis=1)
data.columns = ["Age","Binarizer"]

display(data)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 分段
# MAGIC 
# MAGIC 将连续型变量划分为分类变量。

# COMMAND ----------

# MAGIC %md
# MAGIC ##### KBinsDiscretizer
# MAGIC 
# MAGIC | 参数 | 含义&输入 |
# MAGIC | ----- | ----- |
# MAGIC | n_bins | 每个特征中分箱的个数，默认5，一次会被运用到所有导入的特征 |
# MAGIC | encode| 编码的方式，默认"onehot"<br>"onehot"：做哑变量，之后返回一个稀疏矩阵，每一列是一个特征中的一个类别，含有该类别的样本表示为1，不含的表示为0<br>"ordinal"：每个特征的每个箱都被编码为一个整数，返回每一列是一个特征，每个特征下含有不同整数编码的箱的矩阵<br>"onehot-dense"：做哑变量，之后返回一个密集数组。 |
# MAGIC | strategy| 用来定义箱宽的方式，默认"quantile"<br>"uniform"：表示等宽分箱，即每个特征中的每个箱的最大值之间的差为(特征.max() - 特征.min())/(n_bins)<br>"quantile"：表示等位分箱，即每个特征中的每个箱内的样本数量都相同<br>"kmeans"：表示按聚类分箱，每个箱中的值到最近的一维k均值聚类的簇心得距离都相同。 |

# COMMAND ----------

from sklearn.preprocessing import KBinsDiscretizer

taitanic_train_kbins = taitanic_train.copy()

X = taitanic_train_kbins.iloc[:,5].values.reshape(-1,1)
transformer = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='quantile').fit_transform(X)

data = pd.concat([pd.DataFrame(X), pd.DataFrame(transformer)],axis=1)
data.columns = ["Age","Binarizer"]

display(data)

# COMMAND ----------

# MAGIC %md
# MAGIC # 特征工程
# MAGIC 
# MAGIC **特征工程(Feature Engineering)** 是将原始数据转化成更好的表达问题本质的特征的过程，使得将这些特征运用到预测模型中能提高对不可见数据的模型预测精度。 特征工程简单讲就是发现对因变量有明显影响作用的特征，通常称自变量为特征，特征工程的目的是发现重要特征。如何能够分解和聚合原始数据，以更好的表达问题的本质？这是做特征工程的目的。
# MAGIC 
# MAGIC 本质上来说，呈现给算法的数据应该能拥有基本数据的相关结构或属性。
# MAGIC 
# MAGIC 做特征工程时，其实是将数据属性转换为数据特征的过程。属性代表了数据的所有维度，再数据建模时，如果对原始数据的所有属性进行学习，并不能很好的找到数据的潜在趋势，而通过特征工程对数据进行预处理的话，算法模型能够减少受到噪声的干扰，这样就能够更好的找出趋势。好的特征甚至能够帮助实现使用简单的模型达到很好的效果。
# MAGIC 
# MAGIC | 特征提取（feature extraction） | 特征创造（feature creation） | 特征选择（feature selection） |
# MAGIC | ----- | ----- | ----- |
# MAGIC | 从文字、图像、声音等其他非结构化数据中提取新信息作为特征。比如，通过商品名称提取类别、颜色等。 | 把现有特征进行组合，或互相计算，得到新的特征。比如，如果有速度特征、距离特征，我们就可以计算一个所需时间的特征。 | 从所有的特征中，选择出有意义、对模型有帮助的特征，以避免必须将所有特征都导入模型去训练。 |
# MAGIC 
# MAGIC 一定要抓住给你提供数据的人，尤其时理解业务和数据含义的人，多跟他们沟通，确保你和业务人员一样**理解数据**。
# MAGIC 
# MAGIC 特征选择的第一步，其实是根据我们的目标，用业务常识来选择特征。
# MAGIC 
# MAGIC 比如在泰坦尼克号数据集中，是否存活是我们的标签，很明显，以判断“是否存活”为目的，票号、登船的地方、乘客的编号明显是无关特征，可以直接删除。姓名、舱位等级、船舱编号，也基本可以判断是相关性比较低的特征。性别、年龄、兄弟姐妹的数量、父母和小孩的数量，这些应该是相关性比较高的特征。
# MAGIC 
# MAGIC 所以，**特征工程的第一步，是理解业务、理解数据**。
# MAGIC 
# MAGIC 当然，在真正的数据应用领域，我们的数据的特征很多，并且特征与我们的目标的相关性并不很明显。
# MAGIC 
# MAGIC 如果遇到极端情况，我们无法依赖对业务的理解来选择特征，该怎么办呢？
# MAGIC 
# MAGIC 我们有四种方法可以用来选择特征：过滤法、嵌入法、包装法，和降维算法。

# COMMAND ----------

import zipfile
import pandas as pd

digit_recognizer_train = pd.read_csv("../../Datasets/digit_recognizer_train.zip",compression='zip')

X = digit_recognizer_train.iloc[:,1:]
y = digit_recognizer_train.iloc[:,0]

X.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 过滤法
# MAGIC 
# MAGIC 过滤法通常用作预处理步骤，特征选择完全独立于任何机器学习算法。它是根据各种统计检验中的分数以及相关性的各项指标来选择特征。
# MAGIC 
# MAGIC 全部特征 -> 最佳特征子集 -> 算法 -> 模型评估

# COMMAND ----------

# MAGIC %md
# MAGIC ### 方差过滤

# COMMAND ----------

# MAGIC %md
# MAGIC #### VariabceThreshold
# MAGIC 
# MAGIC 这是通过特征本身的方差来筛选特征的类。比如一个特征本身的方差很小，就表示样本在这个特征上基本没有差异，可能特征中的大多数值都一样，甚至整个特征的取值都相同，那这个特征对于样本区分没有什么作用。**所以无论接下来的特征工程要做什么，都要优先消除方差为0的特征**。VarianceThreshold 有重要参数 **threshold** ，表示方差的阈值，表示舍弃所有方差小于 threshold 的特征，不填的话默认是0，即删除所有的记录都相同的特征。

# COMMAND ----------

from sklearn.feature_selection import VarianceThreshold

selector = VarianceThreshold()
X_0 = selector.fit_transform(X)

X_0.shape, pd.DataFrame(X_0).head()

# COMMAND ----------

# MAGIC %md
# MAGIC 可以看见，我们已经删除了方差为0的特征，但依然剩下了708个特征，明显还需要进一步的特征选择。然而，如果我们指知道我们需要多少个特征，方差也可以帮助我们将特征选择一步到位。比如我们希望留下一半的特征，那可以设定一个让特征总数减半的方差阈值，只要找到方差的中位数，再将这个中位数作为参数 threshold 的值传入就好了，以下代码将784个特征直接缩减到392个：

# COMMAND ----------

import numpy as np

print(np.median(X.var().values))

X_1 = VarianceThreshold(np.median(X.var().values)).fit_transform(X)

X_1.shape

# COMMAND ----------

# MAGIC %md
# MAGIC 当特征是二分类时，特征的取值就是**伯努利随机变量**，这些变量的方差可以计算为：
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=Var%5BX%5D=p%28+1-p%29)
# MAGIC 
# MAGIC 其中![](https://www.zhihu.com/equation?tex=X)时特征矩阵，![](https://www.zhihu.com/equation?tex=p)是二分类特征中的一类在这个特征中所占的概率。

# COMMAND ----------

# 若特征是伯努利随机变量，假设 p=0.8，即二分类特征中某种分类占到 80% 以上的时候删除特征
X_2 = VarianceThreshold(0.8 * (1 - 0.8)).fit_transform(X)
X_2.shape

# COMMAND ----------

# MAGIC %md
# MAGIC #### 方差过滤对模型的影响
# MAGIC 
# MAGIC 我们这样做了以后，多模型效果会有怎样的影响呢？
# MAGIC 
# MAGIC 这里我们通过KNN和随机森林分别在方差过滤前后运行的效果和时间来做比对。

# COMMAND ----------

from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import cross_val_score
import numpy as np
import zipfile
import pandas as pd

digit_recognizer_train = pd.read_csv("../../Datasets/digit_recognizer_train.zip",compression='zip')

X = digit_recognizer_train.iloc[:,1:]
y = digit_recognizer_train.iloc[:,0]

X_fsvar = VarianceThreshold(np.median(X.var().values)).fit_transform(X)

X.shape, X_fsvar.shape

# COMMAND ----------

# MAGIC %md
# MAGIC ##### KNN 方差过滤前

# COMMAND ----------

cross_val_score(KNN(), X, y, cv=5).mean()

# COMMAND ----------

# MAGIC %%timeit
# MAGIC 
# MAGIC cross_val_score(KNN(), X, y, cv=5).mean()

# COMMAND ----------

# MAGIC %md
# MAGIC ##### KNN 方差过滤后

# COMMAND ----------

cross_val_score(KNN(), X_fsvar, y, cv=5).mean()

# COMMAND ----------

# MAGIC %%timeit
# MAGIC 
# MAGIC cross_val_score(KNN(), X_fsvar, y, cv=5).mean()

# COMMAND ----------

# MAGIC %md
# MAGIC ##### RFC 方差过滤前

# COMMAND ----------

cross_val_score(RFC(n_estimators=10, random_state=0), X, y, cv=5).mean()

# COMMAND ----------

# MAGIC %%timeit
# MAGIC 
# MAGIC cross_val_score(RFC(n_estimators=10, random_state=0), X, y, cv=5).mean()

# COMMAND ----------

# MAGIC %md
# MAGIC ##### RFC 方差过滤后

# COMMAND ----------

cross_val_score(RFC(n_estimators=10, random_state=0), X_fsvar, y, cv=5).mean()

# COMMAND ----------

# MAGIC %%timeit
# MAGIC 
# MAGIC cross_val_score(RFC(n_estimators=10, random_state=0), X_fsvar, y, cv=5).mean()

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 结论
# MAGIC 
# MAGIC 为什么随机森林运行如此之快？为什么方差过滤对随机森林没有很大的影响？
# MAGIC 
# MAGIC 这是由于两种算法的原理中涉及到的计算量不同。最近邻算法KNN，单棵决策树，支持向量机SVM，神经网络，回归算法，都需要遍历特征或升维来进行运算，所以他们本身的运算量就很大，需要的时间就很长，因此方差过滤这样的特征选择对他们来说就尤为重要。但对于不需要遍历特征的算法，比如随机森林，它随机选取特征进行分枝，本身运算就非常快速，因此特征选择对它来说效果一般。这其实很容易理解，无论过滤法如何降低特征的数量，随机森林也只会选取固定数量的特征来建模，而最近邻算法就不同了，特征越少，距离计算的维度就越少，模型明显会随着特征的减少变得轻量。
# MAGIC 
# MAGIC 因此，过滤法的**主要对象**是：**需要遍历特征或升维的算法们**，而过滤法的**主要目的**是：**在维持算法表现的前提下，帮助算法们降低计算成本**。

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 思考
# MAGIC 
# MAGIC **过滤法对随机森林无效，却对树模型有效？**
# MAGIC 
# MAGIC 从算法原理来说，传统决策树需要遍历所有特征，计算不纯度后进行分枝，而随机森林却是随机选择特征进行计算和分枝，因此随机森林的运算更快，过滤法对随机森林无用，对决策树有用。
# MAGIC 
# MAGIC 在sklearn中，决策树和随机森林都是随机选择特征进行分枝，但决策树在建模过程中随机抽取的特征数目却远远超过随机森林当中每棵树随机抽取的特征数目。因此，过滤法对随机森林无用，对决策树有用。
# MAGIC 
# MAGIC 也因此，在sklearn中，随机森林中的每棵树都比单独的一棵决策树简单得多，高维数据下的随机森林的计算比决策树快很多。
# MAGIC 
# MAGIC 对受影响的算法来说：
# MAGIC 
# MAGIC | | 阈值很小，被过滤掉的特征比较少 | 阈值较大，被过滤掉的特征有很多 |
# MAGIC | ----- | ----- | ----- |
# MAGIC | 模型表现 | 不会有太大影响 | 可能变更好，代表被过滤掉的特征大部分是噪音；也可能变更糟糕，代表被过滤掉的特征中很多都是有效特征 |
# MAGIC | 运行时间 | 可能降低模型的运行时间，基于方差很小的特征有多少，当方差很小的特征不多时，对模型没有太大影响 | 一定能够降低模型的运行时间，算法在遍历特征时的计算越复杂，运行时间下降得越多 |
# MAGIC 
# MAGIC 在我们的对比中，我们使用的方差阈值是特征方差的中位数，因此属于阈值比较大，过滤掉的特征比较多的情况。我们可以观察到，无论是KNN还是随机森林，在过滤掉一半特征后，模型的精确度都上升了，这说明被我们过滤掉的特征在当前随机模式下大部分是噪音。那我们就可以保留这个去掉一半特征的数据，来为之后的特征选择做准备。当然，如果过滤之后的模型的效果反而变差了，我们就可以认为，被我们过滤掉的特征中有很多都是有效特征，那我们就放弃过滤，使用其他手段来进行特征选择。

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 思考
# MAGIC 
# MAGIC **我们怎么知道，方差过滤掉的到底是噪音还是有效特征呢？过滤后的模型到底会变好还是会变坏呢？**
# MAGIC 
# MAGIC 每个数据集都不一样，只能自己尝试。这里的方差阈值，其实相当于是一个超参数，要选定最优的超参数，我们可以画学习曲线，找模型效果最好的点。但现实中，我们往往不会这样去做，因为这样会耗费大量的时间。我们只会使用阈值为0或者阈值很小的方差过滤，来为我们优先消除一些明显用不到的特征，然后我们会选择更优的特征选择方法继续削减特征数量。

# COMMAND ----------

# MAGIC %md
# MAGIC ### 相关性过滤
# MAGIC 
# MAGIC 我们希望挑选出与标签相关且有意义的特征，因为这样的特征能够为我们提供大量的信息。如果特征与模型无关，那只会浪费我们的计算资源，并且还可能带来噪音。
# MAGIC 
# MAGIC 有三种常用的方法来评判特征与标签之间的相关性：卡方、F检验、互信息。

# COMMAND ----------

# MAGIC %md
# MAGIC #### 卡方过滤
# MAGIC 
# MAGIC 卡方过滤是专门针对离散型标签（即分类问题）的相关性过滤。卡方检验类feature_selection.chi2计算每个非负特征和标签之间的卡方统计量，并依照卡方统计量由高到低为特征排名。再结合feature_selection.SelectKBset这个可以输入“评分标准”来选出前K个分数最高的特征的类，我们可以借此除去最可能独立于标签、与我们分类目的无关的特征。
# MAGIC 
# MAGIC 如果卡方检验检测到某个特征中所有的值都相同，会提示我们使用方差先进行方差过滤。

# COMMAND ----------

from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2
from sklearn.model_selection import cross_val_score
import numpy as np
import zipfile
import pandas as pd

digit_recognizer_train = pd.read_csv("../../Datasets/digit_recognizer_train.zip",compression='zip')

X = digit_recognizer_train.iloc[:,1:]
y = digit_recognizer_train.iloc[:,0]

X_fsvar = VarianceThreshold(np.median(X.var().values)).fit_transform(X)

X.shape, X_fsvar.shape

# COMMAND ----------

# 假设我们需要300个特征
X_fschi = SelectKBest(chi2, k=300).fit_transform(X_fsvar, y)
X_fschi.shape

# COMMAND ----------

# 验证一下模型效果
cross_val_score(RFC(n_estimators=10, random_state=0), X_fschi, y, cv=5).mean()

# COMMAND ----------

# MAGIC %md
# MAGIC 可以看出，模型的效果降低了，这说明我们在设定k=300的时候删除了与模型相关且有效的特征，我们的K值设置得太小，要么我们需要调整K值，要么我们必须放弃相关性过滤。如果模型的表现是提升的，我们就保留相关性过滤的结果。
# MAGIC 
# MAGIC 那么，我们怎么能够知道K值该取多少合适？下面还是用学习曲线来看看结果。

# COMMAND ----------

import matplotlib.pyplot as plt

score = []
for i in range(390, 200, -10):
    X_fschi = SelectKBest(chi2, k=i).fit_transform(X_fsvar, y)
    score.append(cross_val_score(RFC(n_estimators=10, random_state=0), X_fschi, y, cv=5).mean())

plt.plot(range(390, 200, -10), score)

# COMMAND ----------

# MAGIC %md
# MAGIC 通过这条曲线，我们可以观察到，随着K值的不断增加，模型的表现不断上升，这说明数据中所有特征都与标签相关。
# MAGIC 
# MAGIC 学习曲线运行的时间非常长，我们可以通过其他更好的方法来选择K值。

# COMMAND ----------

# MAGIC %md
# MAGIC #### 卡方检验
# MAGIC 
# MAGIC **卡方检验**就是统计样本的实际观测值与理论推断值之间的偏离程度，实际观测值与理论推断值之间的偏离程度就决定卡方值的大小，如果卡方值越大，二者偏差程度越大；反之，二者偏差越小；若两个值完全相等时，卡方值就为0，表明理论值完全符合。
# MAGIC 
# MAGIC **卡方检验**是以![](https://www.zhihu.com/equation?tex=%5Cchi%5E2)分布为基础的一种常用假设检验方法，它的无效假设![](https://www.zhihu.com/equation?tex=H_0)是：**观察频数与期望频数没有差别**。该检验的基本思想是：首先假设![](https://www.zhihu.com/equation?tex=H_0)成立，基于此前提计算出![](https://www.zhihu.com/equation?tex=%5Cchi%5E2)值，它表示观察值与理论值之间的偏离程度。根据![](https://www.zhihu.com/equation?tex=%5Cchi%5E2)分布及自由度可以确定在![](https://www.zhihu.com/equation?tex=H_0)假设成立的情况下获得当前统计量及更极端情况的概率![](https://www.zhihu.com/equation?tex=P)。如果当前统计量大于![](https://www.zhihu.com/equation?tex=P)值，说明观察值与理论值偏离程度太大，应当拒绝无效假设，表示比较资料之间有显著差异；否则就不能拒绝无效假设，尚不能认为样本所代表的实际情况和理论假设有差别。
# MAGIC 
# MAGIC 卡方值计算公式：
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=%5Cchi%5E2%3D%5Csum_%7Bi%3D1%7D%5En%5Cfrac%7B%28real-theory%29%5E2%7D%7Btheory%7D)
# MAGIC 
# MAGIC 卡方检验是以![](https://www.zhihu.com/equation?tex=%5Cchi%5E2)分布为基础的一种常用假设检验方法，![](https://www.zhihu.com/equation?tex=%5Cchi%5E2)分布，就叫做**卡方分布**。
# MAGIC 
# MAGIC 卡方检验的本质是推测两组数据之间的差异，其检验的原假设是“两组数据是相互独立的”。卡方检验返回卡方值和![](https://www.zhihu.com/equation?tex=P)值两个统计量，其中卡方值很难界定有效范围，但![](https://www.zhihu.com/equation?tex=P)值，我们一般使用0.01或0.05作为显著性水平，即![](https://www.zhihu.com/equation?tex=P)值判断的边界。
# MAGIC 
# MAGIC | P值 | <= 0.05或0.01 | > 0.05或0.01 |
# MAGIC | ----- | ----- | ----- |
# MAGIC | 数据差异 | 差异不是自然形成的 | 这些差异是很自然的样本误差 |
# MAGIC | 相关性 | 两组数据是相关的 | 两组数据是相互独立的 |
# MAGIC | 原假设 | 拒绝原假设，接受备择假设 | 接受原假设 |
# MAGIC 
# MAGIC 从特征工程的角度，我们希望选取卡方值很大、![](https://www.zhihu.com/equation?tex=P)值小于0.05的特征，即和标签是相关联的特征。
# MAGIC 
# MAGIC 我们可以直接从chi2实例化后的模型中获得各个特征所对应的卡方值和![](https://www.zhihu.com/equation?tex=P)值。

# COMMAND ----------

chivalue, pvalues_chi = chi2(X_fsvar, y)

# K值取多少？
# 特征个数 - P值大于0.05的特征的个数
K = chivalue.shape[0] - (pvalues_chi > 0.05).sum()

chivalue, pvalues_chi, K

# COMMAND ----------

# MAGIC %md
# MAGIC 可以观察到，所有特征的P值都是0，这说明对于digit recognizor这个数据集来说，方差检验已经把所有和标签无关的特征都剔除了，或者这个数据集本身就不含有与标签无关的特征。在这种情况下，舍弃任何一个特征，都会舍弃对模型有用的信息，而使模型表现下降。因此我们可以衡量计算速度与模型表现，酌情删除一些特征。
# MAGIC 
# MAGIC 当然，我们还可以通过其他的一些验证方法来验证数据集中特征与标签的相关性。

# COMMAND ----------

# MAGIC %md
# MAGIC #### 方差齐性检验
# MAGIC 
# MAGIC **方差齐性**是方差分析和一些均数比较![](https://www.zhihu.com/equation?tex=t)检验的重要前提，**利用![](https://www.zhihu.com/equation?tex=F)检验进行方差齐性检验是最原始的**，但对数据要求比较高。
# MAGIC 
# MAGIC 要求：**样本来自两个独立的、服从正态分布的总体**。
# MAGIC 
# MAGIC **检验原理**
# MAGIC 
# MAGIC 记两独立总体为：
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=X_1%5Csim+N%28%5Cmu_1%2C%5Csigma_1%5E2%29%2C%5Cquad+X_2%5Csim+N%28%5Cmu_2%2C%5Csigma_2%5E2%29)
# MAGIC 
# MAGIC 从两总体中抽取的样本为：
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=X_%7B1i%7D%28i%3D1%2C2%2C%5Ccdots%2Cn_1%29%2C%5Cquad+X_%7B2j%7D%28j%3D1%2C2%2C%5Ccdots%2Cn_2%29)
# MAGIC 
# MAGIC 定义样本均值和样本方差：
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=%5Cbar+X_1%3D%5Cfrac%7B1%7D%7Bn_1%7D%5Csum_%7Bi%3D1%7D%5E%7Bn_1%7DX_%7Bi1%7D%2C%5Cquad+s_1%5E2%3D%5Cfrac%7B1%7D%7Bn_1-1%7D%5Csum_%7Bi%3D1%7D%5E%7Bn_1%7D%28X_%7Bi1%7D-%5Cbar+X_1%29%5E2)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=%5Cbar+X_2%3D%5Cfrac%7B1%7D%7Bn_2%7D%5Csum_%7Bi%3D1%7D%5E%7Bn_2%7DX_%7Bi2%7D%2C%5Cquad+s_2%5E2%3D%5Cfrac%7B1%7D%7Bn_2-1%7D%5Csum_%7Bi%3D1%7D%5E%7Bn_2%7D%28X_%7Bi2%7D-%5Cbar+X_2%29%5E2)
# MAGIC 
# MAGIC 方差齐性双侧检验的原假设和备择假设：
# MAGIC 
# MAGIC > ![](https://www.zhihu.com/equation?tex=H_0%3A%5Csigma_1%5E2%3D%5Csigma_2%5E2)，即两总体方差相等  
# MAGIC ![](https://www.zhihu.com/equation?tex=H_1%3A%5Csigma_1%5E2%5Cneq%5Csigma_2%5E2)，即两总体方差不等
# MAGIC 
# MAGIC 由![](https://www.zhihu.com/equation?tex=F)分布的构造定义：
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=%5Cfrac%7Bs_1%5E2%2F%5Csigma_1%5E2%7D%7Bs_2%5E2%2F%5Csigma_2%5E2%7D%5Csim+F%28n_1-1%2Cn_2-1%29)
# MAGIC 
# MAGIC 其中![](https://www.zhihu.com/equation?tex=n_1-1)、![](https://www.zhihu.com/equation?tex=n_2-1)分别为**分子自由度**和**分母自由度**。
# MAGIC 
# MAGIC 在![](https://www.zhihu.com/equation?tex=H_0)成立的条件下，即![](https://www.zhihu.com/equation?tex=%5Csigma_1%5E2%3D%5Csigma_2%5E2)成立的条件下：
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=%5Cfrac%7Bs_1%5E2%7D%7Bs_2%5E2%7D%5Csim+F%28n_1-1%2Cn_2-1%29)
# MAGIC 
# MAGIC 一般约定取较大的方差作为分子，较小的方差作为分母，这样计算出来的![](https://www.zhihu.com/equation?tex=F%3E1)，缩小了范围，便于查表做出结论。给定显著性水平![](https://www.zhihu.com/equation?tex=%5Calpha)，利用样本数据计算统计量![](https://www.zhihu.com/equation?tex=F_1%3D%5Cfrac%7Bs_1%5E2%7D%7Bs_2%5E2%7D)，若![](https://www.zhihu.com/equation?tex=F_1%3EF_%7B%5Calpha%2C%28n_1-1%2Cn_2-1%29%7D)，这在一次抽样中几乎是不可能发生的（其发生的可能性为![](https://www.zhihu.com/equation?tex=p)值）此时拒绝原假设，认为方差不齐，否则就不拒绝原假设（即认为方差齐）。
# MAGIC 
# MAGIC 对于单侧检验：
# MAGIC 
# MAGIC > ![](https://www.zhihu.com/equation?tex=H_0%3A%5Csigma_1%5E2%3C%5Csigma_2%5E2)  
# MAGIC ![](https://www.zhihu.com/equation?tex=H_1%3A%5Csigma_1%5E2%5Cgeq%5Csigma_2%5E2)
# MAGIC 
# MAGIC 若利用样本计算出来的统计量![](https://www.zhihu.com/equation?tex=%5Cfrac%7Bs_1%5E2%7D%7Bs_2%5E2%7D%3DF_2%3EF_%7B%5Calpha%2C%28n_1-1%2Cn_2-1%29%7D)，则拒绝原假设，否则不拒绝原假设。
# MAGIC 
# MAGIC 对于单侧检验：
# MAGIC 
# MAGIC > ![](https://www.zhihu.com/equation?tex=H_0%3A%5Csigma_2%5E2%3C%5Csigma_1%5E2)  
# MAGIC ![](https://www.zhihu.com/equation?tex=H_1%3A%5Csigma_2%5E2+%5Cgeq+%5Csigma_1%5E2)
# MAGIC 
# MAGIC 若![](https://www.zhihu.com/equation?tex=%5Cfrac%7Bs_1%5E2%7D%7Bs_2%5E2%7D%3DF3%3C%7BF_%7B1-%5Calpha%2C%28n_1-1%2Cn_2-1%29%7D%7D)，则拒绝原假设，否则不拒绝原假设。
