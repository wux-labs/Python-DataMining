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
# MAGIC ### 数值型特征
# MAGIC 
# MAGIC 数值型的幅度变换：对数变换、多项式变换。
# MAGIC * apply + numpy
# MAGIC * preprocessing scaler
# MAGIC 
# MAGIC 对数变换，使得特征和目标变量之间的关系更接近线性，从而提高预测效果。

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

# COMMAND ----------

# MAGIC %md
# MAGIC # 特征工程
# MAGIC 
# MAGIC **特征工程(Feature Engineering)** 是将原始数据转化成更好的表达问题本质的特征的过程，使得将这些特征运用到预测模型中能提高对不可见数据的模型预测精度。 特征工程简单讲就是发现对因变量有明显影响作用的特征，通常称自变量为特征，特征工程的目的是发现重要特征。如何能够分解和聚合原始数据，以更好的表达问题的本质？这是做特征工程的目的。
# MAGIC 
# MAGIC 本质上来说，呈现给算法的数据应该能拥有基本数据的相关结构或属性。
# MAGIC 
# MAGIC 做特征工程时，其实是将数据属性转换为数据特征的过程。属性代表了数据的所有维度，再数据建模时，如果对原始数据的所有属性进行学习，并不能很好的找到数据的潜在趋势，而通过特征工程对数据进行预处理的话，算法模型能够减少受到噪声的干扰，这样就能够更好的找出趋势。好的特征甚至能够帮助实现使用简单的模型达到很好的效果。
