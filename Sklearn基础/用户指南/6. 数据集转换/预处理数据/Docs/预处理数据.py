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
# MAGIC # 标准化，或均值去除和方差缩放
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
mean, np.mean(X_train[:, 0]), np.sqrt(((1 - mean)**2 + (2 - mean)**2 + (0 - mean)**2)/3)

# COMMAND ----------

mean = np.mean(X_train[:, 1])
mean, np.sqrt(((-1 - mean)**2 + (0 - mean)**2 + (1 - mean)**2)/3), np.sqrt(np.var(X_train[:, 1]))

# COMMAND ----------

np.mean(X_train[:, 2]), np.sqrt(np.var(X_train[:, 2]))

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
# MAGIC ## StandardScaler
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
# MAGIC ### 参数

# COMMAND ----------

# MAGIC %md
# MAGIC #### copy
# MAGIC 
# MAGIC **bool, default=True**
# MAGIC 
# MAGIC 如果为 False，尝试避免复制，而是改为进行就地缩放。这不能保证总是在原地工作；例如，如果数据不是NumPy数组或scipy.sparse CSR矩阵，则仍可能返回副本。

# COMMAND ----------

# MAGIC %md
# MAGIC #### with_mean
# MAGIC 
# MAGIC **bool, default=True**
# MAGIC 
# MAGIC 如果为 True，则在缩放之前将数据中心化。当尝试对稀疏矩阵进行时，这不起作用（并且会引发异常），因为将它们中心化需要构建一个密集的矩阵，在常见用例中，该矩阵可能太大而无法放入内存中。

# COMMAND ----------

# MAGIC %md
# MAGIC #### with_std
# MAGIC 
# MAGIC **bool, default=True**
# MAGIC 
# MAGIC 如果为 True，则将数据缩放为单位方差（或等效地，单位标准差）。

# COMMAND ----------

# MAGIC %md
# MAGIC ### 属性

# COMMAND ----------

# MAGIC %md
# MAGIC #### scale_
# MAGIC 
# MAGIC **ndarray of shape (n_features,) or None**
# MAGIC 
# MAGIC 对每个特征的数据进行相对缩放，以实现零均值和单位方差。通常，这是使用 np.sqrt(var_) 计算的。如果方差为零，则我们无法实现单位方差，并且数据保持原样，给出的比例因子为 1。 当 with_std=False，scale_ 等于 None。
# MAGIC 
# MAGIC > 版本 0.17 中的新功能：scale_

# COMMAND ----------

# MAGIC %md
# MAGIC #### mean_
# MAGIC 
# MAGIC **ndarray of shape (n_features,) or None**
# MAGIC 
# MAGIC 训练集中每个特征的平均值。当 with_mean=False 等于 None。

# COMMAND ----------

# MAGIC %md
# MAGIC #### var_
# MAGIC 
# MAGIC **ndarray of shape (n_features,) or None**
# MAGIC 
# MAGIC 训练集中每个特征的方差，用于计算 scale_。当 with_std=False 等于 None。

# COMMAND ----------

# MAGIC %md
# MAGIC #### n_features_in_
# MAGIC 
# MAGIC **int**
# MAGIC 
# MAGIC 在拟合过程中用到的特征数量。
# MAGIC 
# MAGIC > 0.24 版中的新功能。

# COMMAND ----------

# MAGIC %md
# MAGIC #### n_samples_seen_
# MAGIC 
# MAGIC **int or ndarray of shape (n_features,)**
# MAGIC 
# MAGIC 估计器为每个特征处理的样本数。如果没有缺少的样本，则 n_samples_seen 将是一个整数，否则它将是 dtype int 的数组。如果使用 sample_weights，它将是一个浮点数（如果没有丢失的数据）或一个dtype float数组，该数组对到目前为止看到的权重求和。

# COMMAND ----------

# MAGIC %md
# MAGIC ### 方法
# MAGIC 
# MAGIC | 方法 | 描述 |
# MAGIC | ----- | ----- |
# MAGIC | fit(X[, y, sample_weight]) | 计算用于以后缩放的平均值和标准差。 |
# MAGIC | fit_transform(X[, y]) | 拟合数据，然后对其进行转换。 |
# MAGIC | get_params([deep]) | 获取此估计器的参数。 |
# MAGIC | inverse_transform(X[, copy]) | 将数据缩减为原始制图表达。 |
# MAGIC | partial_fit(X[, y, sample_weight]) | 在 X 上在线计算均值和标准值，以便以后进行缩放。 |
# MAGIC | set_params(**params) | 设置此估计器的参数。 |
# MAGIC | transform(X[, copy]) | 通过中心化和缩放执行标准化。 |

# COMMAND ----------

# MAGIC %md
# MAGIC #### fit
# MAGIC 
# MAGIC **fit(X, y=None, sample_weight=None)**
# MAGIC 
# MAGIC **计算平均值和标准差**，用于以后缩放。

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 参数

# COMMAND ----------

# MAGIC %md
# MAGIC ###### X
# MAGIC 
# MAGIC **{array-like, sparse matrix} of shape (n_samples, n_features)**
# MAGIC 
# MAGIC 用于计算平均值和标准差的数据，用于以后沿特征轴进行缩放。

# COMMAND ----------

# MAGIC %md
# MAGIC ###### y
# MAGIC 
# MAGIC **None**
# MAGIC 
# MAGIC 忽视。

# COMMAND ----------

# MAGIC %md
# MAGIC ###### sample_weight
# MAGIC 
# MAGIC **array-like of shape (n_samples,), default=None**
# MAGIC 
# MAGIC 每个样品的单独权重。
# MAGIC 
# MAGIC > 版本 0.24 中的新功能：StandardScaler 支持参数 sample_weight。

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 返回
# MAGIC 
# MAGIC **self：StandardScaler**
# MAGIC 
# MAGIC 标准缩放器。

# COMMAND ----------

# MAGIC %md
# MAGIC #### transform
# MAGIC 
# MAGIC **transform(X, copy=None)**
# MAGIC 
# MAGIC 通过中心化和缩放**执行标准化**。

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 参数

# COMMAND ----------

# MAGIC %md
# MAGIC ###### X
# MAGIC 
# MAGIC **{array-like, sparse matrix of shape (n_samples, n_features)**
# MAGIC 
# MAGIC 用于沿特征轴进行缩放的数据。

# COMMAND ----------

# MAGIC %md
# MAGIC ###### copy
# MAGIC 
# MAGIC **bool, default=None**
# MAGIC 
# MAGIC 是否复制输入 X。

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 返回
# MAGIC 
# MAGIC **X_tr：{ndarray, sparse matrix} of shape (n_samples, n_features)**
# MAGIC 
# MAGIC 转换后的阵列（矩阵或数组）。

# COMMAND ----------

# MAGIC %md
# MAGIC #### fit_transform
# MAGIC 
# MAGIC **fit_transform(X, y=None, \*\*fit_params)**
# MAGIC 
# MAGIC 拟合数据，然后对其进行转换。

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 参数

# COMMAND ----------

# MAGIC %md
# MAGIC ###### X
# MAGIC 
# MAGIC **array-like of shape (n_samples, n_features)**
# MAGIC 
# MAGIC 输入样本。

# COMMAND ----------

# MAGIC %md
# MAGIC ###### y
# MAGIC 
# MAGIC **array-like of shape (n_samples,) or (n_samples, n_outputs), default=None**
# MAGIC 
# MAGIC 目标值（对于无监督转换，None）。

# COMMAND ----------

# MAGIC %md
# MAGIC ###### **fit_params
# MAGIC 
# MAGIC **dict**
# MAGIC 
# MAGIC 其他拟合参数。

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 返回
# MAGIC 
# MAGIC **X_new：ndarray array of shape (n_samples, n_features_new)**
# MAGIC 
# MAGIC 转换后的阵列。

# COMMAND ----------

# MAGIC %md
# MAGIC ## 将特征缩放到某个范围
# MAGIC 
# MAGIC 另一种标准化方法是将特征缩放到给定的最小值和最大值之间，通常介于 0 和 1 之间，或者将每个特征的最大绝对值缩放到单位大小。这可以分别使用 MinMaxScaler 或 MaxAbsScaler 来实现。
# MAGIC 
# MAGIC 使用这种缩放的目的包括实现特征极小方差的鲁棒性以及在稀疏矩阵中保留零元素。
# MAGIC 
# MAGIC 以下是一个将简单的数据矩阵缩放到[0, 1]的示例:

# COMMAND ----------

from sklearn.preprocessing import MinMaxScaler
import numpy as np

X_train = np.array([[ 1., -1.,  2.],
                    [ 2.,  0.,  0.],
                    [ 0.,  1., -1.]])

min_max_scaler = MinMaxScaler()
X_train_minmax = min_max_scaler.fit_transform(X_train)
X_train_minmax

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC 同样的转换实例可以被用与在训练过程中不可见的测试数据：实现和训练数据一致的缩放和移位操作：

# COMMAND ----------

X_test = np.array([[-3., -1.,  4.]])
X_test_minmax = min_max_scaler.transform(X_test)
X_test_minmax

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC 可以检查缩放器（scaler）属性，来观察在训练集中学习到的转换操作的基本性质：

# COMMAND ----------

min_max_scaler.scale_, min_max_scaler.min_

# COMMAND ----------

# MAGIC %md
# MAGIC 如果给 MinMaxScaler 提供一个明确的 feature_range=(min, max) ，完整的公式是：
# MAGIC 
# MAGIC ```python
# MAGIC # 先求出标准化的值
# MAGIC X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
# MAGIC # 再用标准化的值在指定范围内缩放
# MAGIC X_scaled = X_std * (max - min) + min
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC 类 MaxAbsScaler 的工作原理非常相似，但是它只通过除以每个特征的最大值将训练数据特征缩放至 [-1, 1] 范围内，这就意味着，训练数据应该是已经零中心化或者是稀疏数据。
# MAGIC 
# MAGIC 以下是使用上例中数据运用这个缩放器的示例：

# COMMAND ----------

X_train = np.array([[ 1., -1.,  2.],
                    [ 2.,  0.,  0.],
                    [ 0.,  1., -1.]])

max_abs_scaler = preprocessing.MaxAbsScaler()
X_train_maxabs = max_abs_scaler.fit_transform(X_train)
X_train_maxabs

X_test = np.array([[ -3., -1.,  4.]])
X_test_maxabs = max_abs_scaler.transform(X_test)
X_test_maxabs

max_abs_scaler.scale_

# COMMAND ----------

# MAGIC %md
# MAGIC ### MinMaxScaler
# MAGIC 
# MAGIC > class sklearn.preprocessing.MinMaxScaler(feature_range=0, 1, *, copy=True, clip=False)
# MAGIC 
# MAGIC 通过将每个特征缩放到给定范围来转换特征。
# MAGIC 
# MAGIC 该估计器单独缩放和转换每个特征，使其在训练集的给定范围内，例如在0和1之间。
# MAGIC 
# MAGIC 转换由下式给出：
# MAGIC 
# MAGIC ```python
# MAGIC # 先求出标准化的值
# MAGIC X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
# MAGIC # 再用标准化的值在指定范围内缩放
# MAGIC X_scaled = X_std * (max - min) + min
# MAGIC ```
# MAGIC 其中：min, max = feature_range。
# MAGIC 
# MAGIC 此变换通常用作零均值、单位方差缩放的替代方法。

# COMMAND ----------

# MAGIC %md
# MAGIC #### 参数

# COMMAND ----------

# MAGIC %md
# MAGIC ##### feature_range
# MAGIC 
# MAGIC **tuple (min, max), default=(0, 1)**
# MAGIC 
# MAGIC 所需转换数据范围。

# COMMAND ----------

# MAGIC %md
# MAGIC ##### copy
# MAGIC 
# MAGIC **bool, default=True**
# MAGIC 
# MAGIC 设置为 False 可执行就地行规范化并避免复制（如果输入已经是 numpy 数组）。

# COMMAND ----------

# MAGIC %md
# MAGIC ##### clip
# MAGIC 
# MAGIC **bool, default=False**
# MAGIC 
# MAGIC 设置为 True 可将保留数据的转换值裁剪到提供的feature range。
# MAGIC 
# MAGIC > 0.24 版中的新功能。

# COMMAND ----------

# MAGIC %md
# MAGIC #### 属性

# COMMAND ----------

# MAGIC %md
# MAGIC ##### min_
# MAGIC 
# MAGIC **ndarray of shape (n_features,)**
# MAGIC 
# MAGIC 每个特征的最小调整。相当于min - X.min(axis=0) * self.scale_

# COMMAND ----------

# MAGIC %md
# MAGIC ##### scale_
# MAGIC 
# MAGIC **ndarray of shape (n_features,)**
# MAGIC 
# MAGIC 每个特征的数据相对缩放。相当于(max - min) / (X.max(axis=0) - X.min(axis=0))
# MAGIC 
# MAGIC > 版本 0.17 中的新功能：scale_属性。

# COMMAND ----------

# MAGIC %md
# MAGIC ##### data_min_
# MAGIC 
# MAGIC **ndarray of shape (n_features,)**
# MAGIC 
# MAGIC 数据中看到的每个特征的最小值。
# MAGIC 
# MAGIC > 版本 0.17 中的新功能：data_min_属性。

# COMMAND ----------

# MAGIC %md
# MAGIC ##### data_max_
# MAGIC 
# MAGIC **ndarray of shape (n_features,)**
# MAGIC 
# MAGIC 数据中看到的每个特征的最大值。
# MAGIC 
# MAGIC > 版本 0.17 中的新功能：data_max_属性。

# COMMAND ----------

# MAGIC %md
# MAGIC ##### data_range_
# MAGIC 
# MAGIC **ndarray of shape (n_features,)**
# MAGIC 
# MAGIC 数据中看到的每个特征的范围(data_max_ - data_min_)。
# MAGIC 
# MAGIC > 版本 0.17 中的新功能：data_range_属性。

# COMMAND ----------

# MAGIC %md
# MAGIC ##### n_samples_seen_
# MAGIC 
# MAGIC **int or ndarray of shape (n_features,)**
# MAGIC 
# MAGIC 估计器处理的样本数。

# COMMAND ----------

# MAGIC %md
# MAGIC #### 方法
# MAGIC 
# MAGIC | 方法 | 描述 |
# MAGIC | ----- | ----- |
# MAGIC | fit(X[, y]) | 计算要用于以后缩放的最小值和最大值。 |
# MAGIC | fit_transform(X[, y]) | 拟合数据，然后对其进行转换。 |
# MAGIC | get_params([deep]) | 获取此估计器的参数。 |
# MAGIC | inverse_transform(X) | 根据feature_range撤消 X 的缩放。 |
# MAGIC | partial_fit(X[, y]) | 在 X 上在线计算最小值和最大值，以便以后进行缩放。 |
# MAGIC | set_params(**params) | 设置此估计器的参数。 |
# MAGIC | transform(X) | 根据feature_range缩放 X 的特征。 |

# COMMAND ----------

# MAGIC %md
# MAGIC ## 缩放稀疏（矩阵）数据
# MAGIC 
# MAGIC 中心化稀疏(矩阵)数据会破坏数据的稀疏结构，因此很少有一个比较明智的实现方式。但是缩放稀疏输入是有意义的，尤其是当几个特征在不同的量级范围时。
# MAGIC 
# MAGIC MaxAbsScaler 是专门为缩放稀疏数据而设计的，是推荐的执行此操作的方法。但是，StandardScaler 可以接受scipy.sparse矩阵作为输入，只要显式传递with_mean=False给构造函数即可。否则，将引发 ValueError，因为默认的中心化将破坏稀疏性，并且经常会通过无意中分配过多的内存来使执行崩溃。
# MAGIC 
# MAGIC 请注意，缩放器同时接受压缩稀疏行和压缩稀疏列格式（请参见scipy.sparse.csr_matrix和scipy.sparse.csc_matrix）。任何其他稀疏输入都将**转换为压缩稀疏行表示形式**。为避免不必要的内存副本，建议在上游选择 CSR 或 CSC 表示形式。
# MAGIC 
# MAGIC 最后，如果居中数据预计足够小，则使用稀疏矩阵toarray方法将输入显式转换为数组是另一种选择。

# COMMAND ----------

from sklearn.preprocessing import MaxAbsScaler
X = [[ 1., -1.,  2.],
     [ 2.,  0.,  0.],
     [ 0.,  1., -1.]]
transformer = MaxAbsScaler().fit(X)
transformer, transformer.transform(X)

# COMMAND ----------

# MAGIC %md
# MAGIC ### MaxAbsScaler
# MAGIC 
# MAGIC > class sklearn.preprocessing.MaxAbsScaler(*, copy=True)
# MAGIC 
# MAGIC 按每个特征的最大绝对值缩放每个特征。
# MAGIC 
# MAGIC 此估计器单独缩放和转换每个特征，以便训练集中每个特征的最大绝对值为 1.0。它不会移动/中心化数据，因此不会破坏任何稀疏性。
# MAGIC 
# MAGIC 此缩放器还可以应用于稀疏 CSR 或 CSC 矩阵。
# MAGIC 
# MAGIC > 版本 0.17 中的新功能。

# COMMAND ----------

# MAGIC %md
# MAGIC #### 参数

# COMMAND ----------

# MAGIC %md
# MAGIC ##### copy
# MAGIC 
# MAGIC **bool, default=True**
# MAGIC 
# MAGIC 设置为 False 可执行就地行规范化并避免复制（如果输入已经是 numpy 数组）。

# COMMAND ----------

# MAGIC %md
# MAGIC #### 属性

# COMMAND ----------

# MAGIC %md
# MAGIC ##### scale_
# MAGIC 
# MAGIC **ndarray of shape (n_features,)**
# MAGIC 
# MAGIC 每个特征的数据相对缩放。
# MAGIC 
# MAGIC > 版本 0.17 中的新功能：scale_属性。

# COMMAND ----------

# MAGIC %md
# MAGIC ##### max_abs_
# MAGIC 
# MAGIC **ndarray of shape (n_features,)**
# MAGIC 
# MAGIC 每个特征的最大绝对值。

# COMMAND ----------

# MAGIC %md
# MAGIC ##### n_samples_seen_
# MAGIC 
# MAGIC **int**
# MAGIC 
# MAGIC 估计器处理的样本数。

# COMMAND ----------

# MAGIC %md
# MAGIC #### 方法
# MAGIC 
# MAGIC | 方法 | 描述 |
# MAGIC | ----- | ----- |
# MAGIC | fit(X[, y]) | 计算用于以后缩放的最大绝对值。 |
# MAGIC | fit_transform(X[, y]) | 拟合数据，然后对其进行转换。 |
# MAGIC | get_params([deep]) | 获取此估计器的参数。 |
# MAGIC | inverse_transform(X) | 将数据缩减为原始制图表达。 |
# MAGIC | partial_fit(X[, y]) | 在线计算 X 的最大绝对值，以便以后进行缩放。 |
# MAGIC | set_params(**params) | 设置此估计器的参数。 |
# MAGIC | transform(X) | 缩放 X 的特征。 |

# COMMAND ----------

# MAGIC %md
# MAGIC ## 缩放有异常值的数据
# MAGIC 
# MAGIC 如果数据包含许多异常值，则使用数据的均值和方差进行缩放可能不会很好地工作。在这些情况下，您可以使用 RobustScaler 作为直接替换。它对数据的中心和范围使用更可靠的估计值。
# MAGIC 
# MAGIC > 有时候独立地中心化和缩放数据是不够的，因为下游的机器学习模型能够对特征之间的线性依赖做出一些假设(这对模型的学习过程来说是不利的)。要解决此问题，您可以使用 PCA whiten=True 来进一步消除要素之间的线性相关性。

# COMMAND ----------

from sklearn.preprocessing import RobustScaler
X = [[ 1., -2.,  2.],
     [ -2.,  1.,  3.],
     [ 4.,  1., -2.]]
transformer = RobustScaler().fit(X)
transformer, transformer.transform(X)

# COMMAND ----------

# MAGIC %md
# MAGIC ### RobustScaler
# MAGIC 
# MAGIC > class sklearn.preprocessing.RobustScaler(*, with_centering=True, with_scaling=True, quantile_range=25.0, 75.0, copy=True, unit_variance=False)
# MAGIC 
# MAGIC 使用对异常值具有鲁棒性的统计数据来缩放特征。
# MAGIC 
# MAGIC 此缩放器将删除中位数，并根据分位数范围（默认为 IQR：四分位数间距）缩放数据。IQR 是第 1 个四分位数（第 25 个分位数）和第 3 个四分位数（第 75 个分位数）之间的范围。
# MAGIC 
# MAGIC 通过计算训练集中样本的相关统计数据，在每个特征上独立进行中心化和缩放。然后存储中位数和四分位数间范围，以便以在数据上使用transform方法。
# MAGIC 
# MAGIC 数据集的标准化是许多机器学习估计器的共同要求。通常，这是通过删除平均值并缩放到单位方差来完成的。但是，异常值通常可以以负的方式影响样本均值/方差。在这种情况下，中位数和四分位数范围通常能提供更好的结果。
# MAGIC 
# MAGIC > 版本 0.17 中的新功能。

# COMMAND ----------

# MAGIC %md
# MAGIC #### 参数

# COMMAND ----------

# MAGIC %md
# MAGIC ##### with_centering
# MAGIC 
# MAGIC **bool, default=True**
# MAGIC 
# MAGIC 如果为 True，则在缩放之前将数据中心化。当尝试对稀疏矩阵进行transform处理时，这将导致引发异常，因为将它们中心化需要构建一个密集的矩阵，在常见用例中，该矩阵可能太大而无法放入内存中。

# COMMAND ----------

# MAGIC %md
# MAGIC ##### with_scaling
# MAGIC 
# MAGIC **bool, default=True**
# MAGIC 
# MAGIC 如果为 True，则将数据缩放到四分位数范围。

# COMMAND ----------

# MAGIC %md
# MAGIC ##### quantile_range
# MAGIC 
# MAGIC **tuple (q_min, q_max), 0.0 < q_min < q_max < 100.0, default=(25.0, 75.0), == (1st quantile, 3rd quantile), == IQR**
# MAGIC 
# MAGIC 用于计算scale_的分位数范围。
# MAGIC 
# MAGIC > 版本 0.18 中的新功能。

# COMMAND ----------

# MAGIC %md
# MAGIC ##### copy
# MAGIC 
# MAGIC **bool, default=True**
# MAGIC 
# MAGIC 如果为 False，请尝试避免复制，而是改为进行就地缩放。这不能保证总是在原地工作;例如，如果数据不是NumPy数组或scipy.sparse CSR矩阵，则仍可能返回副本。

# COMMAND ----------

# MAGIC %md
# MAGIC ##### unit_variance
# MAGIC 
# MAGIC **bool, default=False**
# MAGIC 
# MAGIC 如果为 True，则缩放数据，使正态分布特征的方差为1。通常，如果标准正态分布的q_max和q_min的x值之间的差值大于1，则数据集将缩小。如果小于1，数据集将被放大。
# MAGIC 
# MAGIC > 0.24 版中的新功能。

# COMMAND ----------

# MAGIC %md
# MAGIC #### 属性

# COMMAND ----------

# MAGIC %md
# MAGIC ##### center_
# MAGIC 
# MAGIC **array of floats**
# MAGIC 
# MAGIC 训练集中每个特征的中位数。

# COMMAND ----------

# MAGIC %md
# MAGIC ##### scale_
# MAGIC 
# MAGIC **array of floats**
# MAGIC 
# MAGIC 训练集中每个特征的（缩放的）四分位数间范围。
# MAGIC 
# MAGIC > 版本 0.17 中的新功能：scale_属性。

# COMMAND ----------

# MAGIC %md
# MAGIC ## 核矩阵的中心化
# MAGIC 
# MAGIC 如果你有一个核矩阵![](https://www.zhihu.com/equation?tex=K)，它计算由函数 ![](https://www.zhihu.com/equation?tex=%5Cphi) 定义的特征空间的点积，那么 KernelCenterer 类能够转化这个核矩阵，通过移除特征空间的平均值，使它包含由函数 ![](https://www.zhihu.com/equation?tex=%5Cphi) 定义的内积。

# COMMAND ----------

from sklearn.preprocessing import KernelCenterer
from sklearn.metrics.pairwise import pairwise_kernels
X = [[ 1., -2.,  2.],
     [ -2.,  1.,  3.],
     [ 4.,  1., -2.]]
K = pairwise_kernels(X, metric='linear')
K

transformer = KernelCenterer().fit(K)
transformer, transformer.transform(K)

# COMMAND ----------

# MAGIC %md
# MAGIC ### KernelCenterer
# MAGIC 
# MAGIC > class sklearn.preprocessing.KernelCenterer
# MAGIC 
# MAGIC 中心化核矩阵。
# MAGIC 
# MAGIC 设 K(x, z) 是由 phi(x)^T phi(z) 定义的核，其中 phi 是映射 x 到希尔伯特空间的函数。KernelCenterer 在不显式计算 phi(x) 的情况下将数据中心化（即，归一化为零均值）。它等效于将 phi(x) 与 sklearn.preprocessing.StandardScaler(with_std=False) 中心化。

# COMMAND ----------

# MAGIC %md
# MAGIC # 非线性变换

# COMMAND ----------

# MAGIC %md
# MAGIC ## 映射到均匀分布

# COMMAND ----------

# MAGIC %md
# MAGIC ## 映射到高斯分布

# COMMAND ----------

# MAGIC %md
# MAGIC # 规范化

# COMMAND ----------

# MAGIC %md
# MAGIC # 编码分类特征

# COMMAND ----------

# MAGIC %md
# MAGIC # 离散化

# COMMAND ----------

# MAGIC %md
# MAGIC # 生成多项式特征
