# Databricks notebook source
# MAGIC %md
# MAGIC # 集成方法

# COMMAND ----------

# MAGIC %md
# MAGIC ## 概述
# MAGIC 
# MAGIC **集成方法**本身不是一个单独的机器学习算法，而是通过在数据上构建多个模型，集成所有模型的建模结果。基本上所有的机器学习领域都可以看到集成方法的身影，在现实中集成方法也有相当大的作用，它可以用来做市场营销模拟的建模，统计客户来源，保留和流失，也可以用来预测疾病的风险和病患者的易感性。
# MAGIC 
# MAGIC 集成方法的**目标是**结合使用给定学习算法构建的多个基本估计器的预测结果，汇总后得到一个综合结果，以此来**获取比单个模型更好的回归或分类表现，以提高单个估计器的泛化性/鲁棒性**。
# MAGIC 
# MAGIC 多个模型集成成为的模型叫做集成评估器，组成集成评估器的每个模型都叫做基评估器。
# MAGIC 
# MAGIC 通常区分两个集合方法族：
# MAGIC 
# MAGIC * **平均方法**，驱动原理是**独立构建多个估计器**，然后平均其预测。平均而言，组合估计器通常优于任何单个基数估计器，因为它的方差减小。  
# MAGIC > 装袋法、随机森林、...
# MAGIC * **提升方法**，基本估计器是**按顺序构建**的，并且尝试减少组合估计器的偏差。动机是将几个弱模型组合在一起，以产生强大的集成。  
# MAGIC > AdaBoost、梯度提升树、...

# COMMAND ----------

# MAGIC %md
# MAGIC ## sklearn中的集成方法

# COMMAND ----------

# MAGIC %md
# MAGIC ### 模块ensemble
# MAGIC 
# MAGIC sklearn中集成方法的类在“ensemble”这个模块下。
# MAGIC 
# MAGIC | 类                          | 描述                                  |
# MAGIC | --------------------------- | ------------------------------------- |
# MAGIC | ensemble.AdaBoostClassifier |  AdaBoost分类 |
# MAGIC | ensemble.AdaBoostRegressor  | AdaBoost回归 |
# MAGIC | ensemble.BaggingClassifier |  装袋分类器 |
# MAGIC | ensemble.BaggingRegressor  | 装袋回归器 |
# MAGIC | ensemble.ExtraTreesClassifier |  Extra-Trees分类 |
# MAGIC | ensemble.ExtraTreesRegressor  | Extra-Trees回归 |
# MAGIC | ensemble.GradientBoostingClassifier |  梯度提升分类 |
# MAGIC | ensemble.GradientBoostingRegressor  | 梯度提升回归 |
# MAGIC | ensemble.IsolationForest | 隔离森林 |
# MAGIC | ensemble.RandomForestClassifier | 随机森林分类 |
# MAGIC | ensemble.RandomForestRegressor  | 随机森林回归 |
# MAGIC | ensemble.RandomTreesEmbedding | 完全随机树的集成 |
# MAGIC | ensemble.StackingClassifier | 具有最终分类器的估计器堆栈 |
# MAGIC | ensemble.StackingRegressor | 具有最终回归器的估计器堆栈 |
# MAGIC | ensemble.VotingClassifier | 用于不适合估计器的软投票/多数规则分类器 |
# MAGIC | ensemble.VotingRegressor | 用于不适合估计器的软投票/多数规则回归器 |
# MAGIC | ensemble.HistGradientBoostingClassifier | 基于直方图的梯度提升分类树 |
# MAGIC | ensemble.HistGradientBoostingRegressor | 基于直方图的梯度提升回归树 |

# COMMAND ----------

# MAGIC %md
# MAGIC ## RandomForestClassifier
# MAGIC 
# MAGIC 随机森林分类器。
# MAGIC 
# MAGIC 随机森林是一种元估计器，它拟合数据集的各个子样本上的多个决策树分类器，并使用平均来提高预测准确性和控制过度拟合。子样本大小由max_samples参数如果bootstrap=True（默认值） 控制，否则将使用整个数据集来构建每个树。
# MAGIC 
# MAGIC * 0.24
# MAGIC > class sklearn.ensemble.RandomForestClassifier(n_estimators=100, *, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, **min_impurity_split=None,** bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None)
# MAGIC 
# MAGIC * 1.0.2
# MAGIC > class sklearn.ensemble.RandomForestClassifier(n_estimators=100, *, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None)
# MAGIC 
# MAGIC 单个决策树的准确率越高，随机森林的准确率也会越高，因为装袋法是依赖于平均值或者少数服从多数的原则来决定集成的结果的。

# COMMAND ----------

# MAGIC %md
# MAGIC ### 参数

# COMMAND ----------

# MAGIC %md
# MAGIC #### n_estimators
# MAGIC 
# MAGIC **int, default=100**
# MAGIC 
# MAGIC 随机森林中树的数量。就是基评估器的数量。
# MAGIC 
# MAGIC 这个参数对随机森林模型的精确性影响是单调的，**n_estimators越大，模型的效果往往越好**。但是相应的，任何模型都有决策边界，n_estimators达到一定的程度之后，随机森林的精确性往往不再上升，或开始波动，并且n_estimators越大，需要的计算量和内存也越大，训练的时间也会越长。
# MAGIC 
# MAGIC 对于这个参数，我们是渴望在训练难度和模型效果之间取得平衡。
# MAGIC 
# MAGIC > 在版本 0.22 中更改： 默认值在 0.22 中从 10 更改为 100。

# COMMAND ----------

# MAGIC %md
# MAGIC #### 不纯度参数

# COMMAND ----------

# MAGIC %md
# MAGIC ##### criterion
# MAGIC 
# MAGIC **{"gini", "entropy"}, default="gini"**
# MAGIC 
# MAGIC 用来衡量分枝质量的指标，即衡量不纯度的指标。
# MAGIC 
# MAGIC 为了要将表格转化为一棵树，决策树需要找出最佳节点和最佳的分枝方法，对分类树来说，衡量这个“最佳”的指标叫做“**不纯度（杂质）**”。通常来说，不纯度越低，决策树对训练集的拟合越好。现在使用的决策树算法在分枝方法上的核心大多是围绕在对某个不纯度相关指标的最优化上。
# MAGIC 
# MAGIC 不纯度基于节点来计算，树中的每个节点都会有一个不纯度，并且子节点的不纯度一定低于父节点的，也就是说，在同一棵决策树上，叶子节点的不纯度一定是最低的。
# MAGIC 
# MAGIC Criterion这个参数正是用来决定不纯度的计算方法的。sklearn提供了两种选择：
# MAGIC 
# MAGIC * gini：使用基尼系数
# MAGIC * entropy：使用信息熵
# MAGIC 
# MAGIC 比起基尼系数，信息熵对不纯度更加敏感，对不纯度的惩罚最强。但在实际使用中，信息熵和基尼系数的效果基本相同。信息熵的计算比基尼系数缓慢一些，因为基尼系数的计算不涉及对数。另外，因为信息熵对不纯度更加敏感，所以信息熵作为指标时，决策树的生长会更加“精细”，因此对于高维数据或者噪音很多的数据，信息熵很容易过拟合，基尼系数在这种情况下效果往往比较好。

# COMMAND ----------

# MAGIC %md
# MAGIC #### 随机性参数

# COMMAND ----------

# MAGIC %md
# MAGIC ##### random_state
# MAGIC 
# MAGIC **int, RandomState instance or None, default=None**
# MAGIC 
# MAGIC 控制构建树时所用样本的随机性（如果 bootstrap=True） 和在每个节点（如果 max_features < n_features）上查找最佳拆分时要考虑的特征的采样。
# MAGIC 
# MAGIC 随机森林的本质是一种袋装集成算法，袋装集成算法是对基评估器的预测结果取平均值原则或少数服从多数的原则来决定集成评估器的结果。随机森林判断错误的概率是：
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=E_%7Brandom%5C%5Fforest%7D%3D%5Csum_%7Bi%3De_%7Bmin%7D%7D%5ENC_N^iε^i%281-ε%29^%7BN-i%7D)
# MAGIC 
# MAGIC 其中，![](https://www.zhihu.com/equation?tex=N)是随机森林中树的数量，![](https://www.zhihu.com/equation?tex=i)是随机森林判断错误的树的数量，![](https://www.zhihu.com/equation?tex=e_%7Bmin%7D)是要判定随机森林判断错误至少需要满足的判断错误的树的数量，根据少数服从多数原则，通常![](https://www.zhihu.com/equation?tex=e_%7Bmin%7D>%5Cfrac%7BN%7D%7B2%7D)，![](https://www.zhihu.com/equation?tex=ε)是一棵树判断错误的概率，![](https://www.zhihu.com/equation?tex=%281-ε%29)是一棵树判断正确的概率，![](https://www.zhihu.com/equation?tex=%7BN-i%7D)是判断正确的次数。
# MAGIC 
# MAGIC 随机森林中的random_state用法与分类树中相似，只不过在分类树中，一个random_state只控制生成一棵树，而随机森林中的random_state控制的是生成森林的模式，而不是让一个森林中只有一棵树。
# MAGIC 
# MAGIC 对于固定的random_state生成的是同一片森林，森林中的每一棵树的random_state是不一样的，但由于生成的是同一片森林，所以多次训练产生的同一片森林中的每一棵树的random_state不会变。
# MAGIC 
# MAGIC 我们可以观察到，当random_state固定时，随机森林中生成的是固定的一片森林，也就是一组固定的树，但每棵树依然是不一样的，这是用“随机挑选特征进行分枝”的方法得到的随机性。并且我们可以证明，当这种随机性越大的时候，装袋法的效果一般会越来越好。**用装袋法集成时，基分类器应当时相互独立的，是不相同的**。
# MAGIC 
# MAGIC 但是这种做法的局限性是很强的，当我们需要成千上万棵树的时候，数据不一定能够提供成千上万的特征来让我们构筑尽量多的不同的树，因此，除了random_state，我们还需要其他的随机性。

# COMMAND ----------

import numpy as np
from scipy.special import comb
import matplotlib.pyplot as plt

x = np.linspace(0,1,20)
y = []

# 森林中树的数量
N = 25
# 少数服从多数原则，要判定随机森林判断错误，至少需要多少棵树判断错误
emin = N - N // 2


for epsilon in x:
    # 在单棵分类树的错误率从0~1的变化过程中，随机森林的判断错误的概率
    E = np.array([comb(N,i)*(epsilon**i)*((1-epsilon)**(N-i)) for i in range(emin,N+1)]).sum()
    y.append(E)

plt.figure(figsize=(20,8))
# 设置字体
# plt.rcParams["font.sans-serif"]=["SimHei"]
# 该语句解决图像中的“-”负号的乱码问题
# plt.rcParams["axes.unicode_minus"]=False

# 如果森林中的每一棵树都是一样的，那么随机森林判断错误的概率等于单棵树的错误率
plt.plot(x,x,'--',color='red',label="如果森林中的树都一样")
# 如果森林中的每一棵树是不一样的，那么随机森林判断错误的概率就适用于上面的公式
plt.plot(x,y,'o-',label="如果森林中的树不一样")

plt.xlabel("个体估计误差")
plt.ylabel("随机森林误差")

plt.legend()

# 可以看到，当单棵树的错误率高于50%的时候，随机森林的错误率要高于单棵树的，也就是模型效果更差
# 所以，随机森林要求单棵树的准确率要超过50%
# 袋装法要求 基分类器的判断准确率至少要超过随机分类器，也就是说，基分类器的判断准确率至少要超过50%

# COMMAND ----------

# MAGIC %md
# MAGIC ##### bootstrap
# MAGIC 
# MAGIC **bool, default=True**
# MAGIC 
# MAGIC 构建树时是否使用引导样本（bootstrap samples）。如果为 False，则使用整个数据集来生成每个树。
# MAGIC 
# MAGIC 要让基分类器尽量都不一样，一种很容易理解的方法是使用不同的训练集来进行训练，而装袋法正是通过有放回的随机抽样技术来形成不同的训练数据，bootstrap就是用来控制抽样技术的参数。
# MAGIC 
# MAGIC 在一个含有n个样本的原始训练集中，我们进行随机采样，每次采样一个样本，并在抽取下一个样本之前将该样本放回到原始训练集，也就是说下次采样时这个样本依然可能被采集到，这样采集n次，最终得到一个和原始训练集一样大，n个样本组成的自助集。由于是随机采样，这样每次的自助集和原始数据集不同，和其他的采样集也是不同的。这样我们就可以自由创造取之不尽用之不竭、并且互不相同的自助集，用这些自助集来训练我们的基分类器，我们的基分类器自然也就各不相同了。
# MAGIC 
# MAGIC **bootstrap默认True，代表采用这种有放回的随机抽样技术。** 通常，这个参数不会被我们设置成False。
# MAGIC 
# MAGIC 然而有放回抽样也会有自己的问题。由于是有放回，一些样本可能在同一个自助集中出现多次，而其他一些却可能被忽略，一般来说，自助集大约平均会包含63%的原始数据。因为每一个样本被抽到某个自助集中的概率为：
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=1-%28+1-%5Cfrac%7B1%7D%7Bn%7D%29^n)
# MAGIC 
# MAGIC 当n足够大的时候，这个概率收敛于![](https://www.zhihu.com/equation?tex=1-%5Cfrac%7B1%7D%7Be%7D)，约等于0.632。因此，会有约37%的训练数据被浪费掉，没有参与建模，这些数据被称为袋外数据（out of bag data，简写为oob）。除了我们最开始就划分好的测试集之外，这些数据也可以被用来作为集成算法的测试集。**也就是说，在使用随机森林时，我们可以不划分测试集和训练集，只需要使用袋外数据来测试我们的模型即可**。当然，这也不是绝对的，当n和n_estimators都不够大的时候，很可能就没有数据掉落在袋外，自然也就无法使用oob数据来测试模型了。

# COMMAND ----------

# MAGIC %md
# MAGIC ##### oob_score
# MAGIC 
# MAGIC **bool, default=False**
# MAGIC 
# MAGIC 是否使用袋外样本来估计泛化精确性。仅当 bootstrap = True 时才可用。
# MAGIC 
# MAGIC 如果希望用袋外数据来测试，则需要在实例化时就将oob_score这个参数调整为True，训练完毕之后，我们可以用随机森林的一个属性 oob_score_ 来查看我们在袋外数据上测试的结果。

# COMMAND ----------

# MAGIC %md
# MAGIC #### 剪枝参数
# MAGIC 
# MAGIC 在不加限制的情况下，一棵决策树会生长到衡量不纯度的指标最优，或者没有更多的特征可用为止，这样的决策树往往会过拟合。
# MAGIC 
# MAGIC 为了让决策树有更好的泛化性，我们需要对决策树进行剪枝。

# COMMAND ----------

# MAGIC %md
# MAGIC ##### max_depth
# MAGIC 
# MAGIC **int, default=None**
# MAGIC 
# MAGIC 树的最大深度。如果为 None，则节点将展开，树辉持续生长，直到所有叶子都是纯的，或者直到所有叶子包含的样本少于 min_samples_split。

# COMMAND ----------

# MAGIC %md
# MAGIC ##### min_samples_split
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

# MAGIC %md
# MAGIC ##### min_samples_leaf
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

# MAGIC %md
# MAGIC ##### max_features
# MAGIC 
# MAGIC **int, float or {"auto", "sqrt", "log2"}, default="auto"**
# MAGIC 
# MAGIC 寻找最佳分枝时要考虑的特征数量：
# MAGIC 
# MAGIC * 如果为 int，则考虑每次分枝时的特征。max_features
# MAGIC * 如果是 float，则为比例，每次分枝时考虑的特征数目是max_featuresint(max_features * n_features)。
# MAGIC * 如果为"auto"，则 .max_features=sqrt(n_features)
# MAGIC * 如果为"sqrt"，则 .max_features=sqrt(n_features)（与 auto 一样）
# MAGIC * 如果为"log2"，则 .max_features=log2(n_features)
# MAGIC * 如果为 None，则 .max_features=n_features
# MAGIC 
# MAGIC > 注意：如果在限制的 max_features 中，决策树无法找到节点样本上至少一个有效的分枝，那对分枝的搜索不会停止，决策树将会检查比限制的 max_features 数目更多的特征。
# MAGIC 
# MAGIC max_features 用来限制在分枝时考虑的特征个数，超过限制个数的特征都会被舍弃。
# MAGIC 
# MAGIC max_features 是用来限制高维度数据的过拟合的剪枝参数，但其方法比较暴力，是直接限制可以使用的特征的数量而强行使决策树停止生长。在不知道决策树中的各个特征的重要性的情况下，强行设定这个参数可能会导致模型学习不足。

# COMMAND ----------

# MAGIC %md
# MAGIC ##### max_leaf_nodes
# MAGIC 
# MAGIC **int, default=None**
# MAGIC 
# MAGIC 最大叶节点数量。在最佳分枝方式下，以max_leaf_nodes为限制来生长树。如果为 None，则叶节点数不受限制。

# COMMAND ----------

# MAGIC %md
# MAGIC ##### min_impurity_decrease
# MAGIC 
# MAGIC **float, default=0.0**
# MAGIC 
# MAGIC 如果节点的分裂导致不纯度的减少大于或等于此值，则节点将被分裂。
# MAGIC 
# MAGIC 加权不纯度减少方程如下：
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=%5Cfrac%7BN_t%7D%7BN%7D*%28+impurity-%5Cfrac%7BN_%7Bt_R%7D%7D%7BN_t%7D*right%5C%5F+impurity-%5Cfrac%7BN_%7Bt_L%7D%7D%7BN_t%7D*left%5C%5F+impurity%29)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=%5Cfrac%7BN_t%7D%7BN%7D*%28+不纯度-%5Cfrac%7BN_%7Bt_R%7D%7D%7BN_t%7D*右侧分枝的不纯度-%5Cfrac%7BN_%7Bt_L%7D%7D%7BN_t%7D*左侧分枝的不纯度%29)
# MAGIC 
# MAGIC 其中![](https://www.zhihu.com/equation?tex=N)是样本总数，![](https://www.zhihu.com/equation?tex=N_t)是当前节点上的样本数，![](https://www.zhihu.com/equation?tex=N_%7Bt_L%7D)是左子级中的样本数，![](https://www.zhihu.com/equation?tex=N_%7Bt_R%7D)是右子级中的样本数。
# MAGIC 
# MAGIC > 注意：如果 sample_weight 在 fit 接口中有值，则![](https://www.zhihu.com/equation?tex=N)、![](https://www.zhihu.com/equation?tex=N_t)、![](https://www.zhihu.com/equation?tex=N_%7Bt_L%7D)、![](https://www.zhihu.com/equation?tex=N_%7Bt_R%7D)都是指样本量的权重，而非单纯的样本数量。
# MAGIC 
# MAGIC > 0.19 版中的新功能。

# COMMAND ----------

# MAGIC %md
# MAGIC ##### min_impurity_split
# MAGIC 
# MAGIC **float, default=None**
# MAGIC 
# MAGIC 早期停止树木生长的阈值。如果节点的不纯度高于阈值，则节点将分裂，否则为叶子。
# MAGIC 
# MAGIC > 自版本 0.19 起弃用：min_impurity_split已弃用，转而支持 0.19 中的min_impurity_decrease。默认值 min_impurity_split 在 0.23 中已从 1e-7 更改为 0，**并将在 1.0 中删除（重命名为 0.25）。** 请改用min_impurity_decrease。

# COMMAND ----------

# MAGIC %md
# MAGIC #### 目标权重参数

# COMMAND ----------

# MAGIC %md
# MAGIC ##### min_weight_fraction_leaf
# MAGIC 
# MAGIC **float, default=0.0**
# MAGIC 
# MAGIC 一个叶节点要存在所需要的权重占输入模型的数据集的总权重的比例。
# MAGIC 
# MAGIC 总权重由fit接口中的sample_weight参数确定，当sample_weight为None时，默认所有样本的权重相同。

# COMMAND ----------

# MAGIC %md
# MAGIC ##### class_weight
# MAGIC 
# MAGIC **{“balanced”, “balanced_subsample”}, dict or list of dicts, default=None**
# MAGIC 
# MAGIC 与标签相关联的权重，表现方式是 {class_label: weight}。如果为 None，则所有类都应具有权重 1。对于多输出问题，可以按与 y 列相同的顺序提供字典列表。
# MAGIC 
# MAGIC 请注意，对于多输出（包括多标签），应在其自己的字典中为每列的每个类定义权重。例如，对于四类多标签分类，权重应为[{0:1, 1:1}, {0:1, 1:5}, {0:1, 1:1}, {0:1, 1:1}]，而不是[{1:1}, {2:5}, {3:1}, {4:1}]。
# MAGIC 
# MAGIC "balanced" 模式使用 y 的值自动调整权重，与输入数据中的类频率成反比，如n_samples / (n_classes * np.bincount(y))
# MAGIC 
# MAGIC 对于多输出，每列 y 的权重将相乘。
# MAGIC 
# MAGIC > 请注意，如果指定了sample_weight，这些权重将乘以sample_weight（通过拟合方法）。

# COMMAND ----------

# MAGIC %md
# MAGIC #### n_jobs
# MAGIC 
# MAGIC **int, default=None**
# MAGIC 
# MAGIC 要并行运行的作业数。 fit, predict, decision_path 和 apply都在树上并行化。 None 表示 1，除非在joblib.parallel_backend上下文中。 -1 表示使用所有处理器。

# COMMAND ----------

# MAGIC %md
# MAGIC #### verbose
# MAGIC 
# MAGIC **int, default=0**
# MAGIC 
# MAGIC 控制拟合和预测时的详细程度。

# COMMAND ----------

# MAGIC %md
# MAGIC #### warm_start
# MAGIC 
# MAGIC **bool, default=False**
# MAGIC 
# MAGIC 设置为 True 时，重用上一个调用的解决方案以拟合并向融合中添加更多估计器，否则，只需拟合一个全新的林。

# COMMAND ----------

# MAGIC %md
# MAGIC #### ccp_alpha
# MAGIC 
# MAGIC **non-negative float, default=0.0**
# MAGIC 
# MAGIC 用于最小成本-复杂性修剪的复杂度参数。将选择的成本复杂度大于将选择的成本复杂度的子树。默认情况下，不执行修剪。
# MAGIC 
# MAGIC > 0.22 版中的新功能。

# COMMAND ----------

# MAGIC %md
# MAGIC #### max_samples
# MAGIC 
# MAGIC **int or float, default=None**
# MAGIC 
# MAGIC 如果 bootstrap 为 True，则表示要从 X 中提取以训练每个基本估计器的样本数。
# MAGIC 
# MAGIC * 如果为"None"（默认值），则绘制X.shape[0]个样本。
# MAGIC * 如果为 int，则绘制max_samples个样本。
# MAGIC * 如果是浮动，则绘制max_samples * X.shape[0]个样本。因此，max_samples应该在间隔(0, 1)内。
# MAGIC 
# MAGIC > 版本 0.22 中的新功能。

# COMMAND ----------

# MAGIC %md
# MAGIC ### 属性
# MAGIC 
# MAGIC 属性是在模型训练之后，能够调用查看的模型的各种性质。

# COMMAND ----------

# MAGIC %md
# MAGIC #### base_estimator_
# MAGIC 
# MAGIC 用于创建拟合子估计器集合的子估计器模板。

# COMMAND ----------

# MAGIC %md
# MAGIC #### estimators_
# MAGIC 
# MAGIC 拟合子估计器的集合。

# COMMAND ----------

# MAGIC %md
# MAGIC #### classes_
# MAGIC 
# MAGIC 类标签（单输出问题）或类标签数组列表（多输出问题）。

# COMMAND ----------

# MAGIC %md
# MAGIC #### n_classes_
# MAGIC 
# MAGIC 类数（单个输出问题），或包含每个输出的类数的列表（多输出问题）。

# COMMAND ----------

# MAGIC %md
# MAGIC #### n_features_
# MAGIC 
# MAGIC 执行fit时的特征数。

# COMMAND ----------

# MAGIC %md
# MAGIC #### n_outputs_
# MAGIC 
# MAGIC 执行fit时的输出数。

# COMMAND ----------

# MAGIC %md
# MAGIC #### feature_importances_
# MAGIC 
# MAGIC 基于不纯度的特征重要性。

# COMMAND ----------

# MAGIC %md
# MAGIC #### oob_score_
# MAGIC 
# MAGIC 使用袋外估计获得的训练数据集的分数。仅当 oob_score 为 True 时，此属性才存在。

# COMMAND ----------

# MAGIC %md
# MAGIC #### oob_decision_function_
# MAGIC 
# MAGIC 使用训练集上的袋外估计计算的决策函数。如果n_estimators很小，则数据点可能在引导期间从未被遗漏。在这种情况下，oob_decision_function_可能包含 NaN。仅当 ob_score 为 True 时，此属性才存在。

# COMMAND ----------

# MAGIC %md
# MAGIC ### 方法
# MAGIC 
# MAGIC | 方法 | 描述 |
# MAGIC | ---- | ---- |
# MAGIC | apply(X) | 将森林中的树应用于 X，返回叶子索引。 |
# MAGIC | decision_path(X) | 返回森林中的决策路径。 |
# MAGIC | fit(X, y[, sample_weight]) | 从训练集 (X, y) 构建一片森林。 |
# MAGIC | get_params([deep]) | 获取此估计器的参数。 |
# MAGIC | predict(X) | 预测 X 的类。 |
# MAGIC | predict_log_proba(X) | 预测 X 的类对数概率。 |
# MAGIC | predict_proba(X) | 预测 X 的类概率。 |
# MAGIC | score(X, y[, sample_weight]) | 返回给定测试数据和标签的平均精度。 |
# MAGIC | set_params(**params) | 设置此估计器的参数。 |

# COMMAND ----------

# MAGIC %md
# MAGIC #### fit
# MAGIC 
# MAGIC **fit(X, y, sample_weight=None)**
# MAGIC 
# MAGIC 从训练集 (X, y) 构建一片森林。

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 参数
# MAGIC 
# MAGIC * **X**: {array-like, sparse matrix} of shape (n_samples, n_features)  
# MAGIC 训练输入示例。在内部，将会转换为dtype=np.float32，如果提供给稀疏矩阵，则将其转换为csc_matrix。
# MAGIC * **y**: array-like of shape (n_samples,) or (n_samples, n_outputs)  
# MAGIC 目标值（分类中的类标签、回归中的实数）。
# MAGIC * **sample_weight**: array-like of shape (n_samples,), default=None  
# MAGIC 样本权重。如果为 None，则样本的权重相等。在每个节点中搜索拆分时，将忽略将创建具有净零或负权重的子节点的拆分。如果拆分会导致任何单个类在任一子节点中承载负权重，则也会忽略拆分。

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 返回
# MAGIC 
# MAGIC * **self**: RandomForestClassifier  
# MAGIC 拟合估计器。

# COMMAND ----------

# MAGIC %md
# MAGIC #### score
# MAGIC 
# MAGIC **score(X, y, sample_weight=None)**
# MAGIC 
# MAGIC 返回给定测试数据和标签的平均精度。
# MAGIC 
# MAGIC 在多标签分类中，这是子集准确性，这是一个苛刻的指标，因为您需要为每个样本正确预测每个标签集。

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 参数
# MAGIC 
# MAGIC * **X**: array-like of shape (n_samples, n_features)  
# MAGIC 测试样本。
# MAGIC * **y**: array-like of shape (n_samples,) or (n_samples, n_outputs)  
# MAGIC X的真实标签。
# MAGIC * **sample_weight**: array-like of shape (n_samples,), default=None  
# MAGIC 样本权重。

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 返回
# MAGIC 
# MAGIC * **score**: float  
# MAGIC 模型预测的平均精确度。

# COMMAND ----------

# MAGIC %md
# MAGIC #### apply
# MAGIC 
# MAGIC **apply(X)**
# MAGIC 
# MAGIC 返回预测每个样本的叶子的索引。

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 参数
# MAGIC 
# MAGIC * **X**: {array-like, sparse matrix} of shape (n_samples, n_features)  
# MAGIC 训练输入示例。在内部，将会转换为dtype=np.float32，如果提供给稀疏矩阵，则将其转换为csr_matrix。

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 返回
# MAGIC 
# MAGIC * **X_leaves**: ndarray of shape (n_samples, n_estimators)
# MAGIC 对于 X 中的每个数据点 x 和森林中的每棵树，返回叶子 x 的索引。

# COMMAND ----------

# MAGIC %md
# MAGIC #### predict
# MAGIC 
# MAGIC **predict(X)**
# MAGIC 
# MAGIC 预测 X 的类。
# MAGIC 
# MAGIC 输入样本的预测类是森林中树木的投票，由其概率估计值加权。也就是说，预测的类是跨树具有最高平均概率估计的类。

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 参数
# MAGIC 
# MAGIC * **X**: {array-like, sparse matrix} of shape (n_samples, n_features)  
# MAGIC 训练输入示例。在内部，将会转换为dtype=np.float32，如果提供给稀疏矩阵，则将其转换为csr_matrix。

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 返回
# MAGIC 
# MAGIC * **y**: ndarray of shape (n_samples,) or (n_samples, n_outputs)  
# MAGIC 预测类。

# COMMAND ----------

# MAGIC %md
# MAGIC ## 袋装法的必要条件
# MAGIC 
# MAGIC 在使用袋装法时，要求：
# MAGIC * **基评估器要尽量独立**
# MAGIC * **基分类器的判断准确率至少要超过随机分类器**，也就是说，基分类器的判断准确率至少要超过50%

# COMMAND ----------

# MAGIC %md
# MAGIC ## RandomForestRegressor
# MAGIC 
# MAGIC 随机森林回归器。
# MAGIC 
# MAGIC 随机森林是一种元估计器，它拟合数据集各个子样本上的许多分类决策树，并使用平均来提高预测准确性和控制过度拟合。子样本大小由max_samples参数如果bootstrap=True（默认值） 控制，否则将使用整个数据集来构建每个树。
# MAGIC 
# MAGIC * 0.24
# MAGIC > class sklearn.ensemble.RandomForestRegressor(n_estimators=100, *, criterion='mse', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, **min_impurity_split=None,** bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False, ccp_alpha=0.0, max_samples=None)
# MAGIC 
# MAGIC * 1.0.2
# MAGIC > class sklearn.ensemble.RandomForestRegressor(n_estimators=100, *, criterion='squared_error', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False, ccp_alpha=0.0, max_samples=None)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 参数

# COMMAND ----------

# MAGIC %md
# MAGIC #### n_estimators
# MAGIC 
# MAGIC **int, default=100**
# MAGIC 
# MAGIC 随机森林中树的数量。就是基评估器的数量。
# MAGIC 
# MAGIC 这个参数对随机森林模型的精确性影响是单调的，**n_estimators越大，模型的效果往往越好**。但是相应的，任何模型都有决策边界，n_estimators达到一定的程度之后，随机森林的精确性往往不再上升，或开始波动，并且n_estimators越大，需要的计算量和内存也越大，训练的时间也会越长。
# MAGIC 
# MAGIC 对于这个参数，我们是渴望在训练难度和模型效果之间取得平衡。
# MAGIC 
# MAGIC > 在版本 0.22 中更改： 默认值在 0.22 中从 10 更改为 100。

# COMMAND ----------

# MAGIC %md
# MAGIC #### 不纯度参数

# COMMAND ----------

# MAGIC %md
# MAGIC ##### criterion

# COMMAND ----------

# MAGIC %md
# MAGIC ###### 0.24
# MAGIC 
# MAGIC **{"mse", "mae"}, default="mse"**
# MAGIC 
# MAGIC 用来衡量分枝质量的指标，即衡量不纯度的指标。
# MAGIC 
# MAGIC 支持：
# MAGIC * mse：均方误差，这等于方差约简作为特征选择标准
# MAGIC * mae：平均绝对误差
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=MSE%3D%5Cfrac%7B1%7D%7BN%7D%5Csum_%7Bi%3D1%7D%5EN%28f_i-y_i%29^2)
# MAGIC 
# MAGIC 其中![](https://www.zhihu.com/equation?tex=N)是样本数量，![](https://www.zhihu.com/equation?tex=i)是每一个数据样本，![](https://www.zhihu.com/equation?tex=f_i)是模型回归出的数值，![](https://www.zhihu.com/equation?tex=y_i)是样本点![](https://www.zhihu.com/equation?tex=i)实际的数值标签。所以![](https://www.zhihu.com/equation?tex=MSE)的本质，其实是样本真实数据与回归结果的差异。
# MAGIC 
# MAGIC **在回归树中，![](https://www.zhihu.com/equation?tex=MSE)不只是我们衡量分枝质量的指标，也是我们最常用的衡量回归树回归质量的指标。**
# MAGIC 
# MAGIC > 版本 0.18 新增：mae 平均绝对误差。 

# COMMAND ----------

# MAGIC %md
# MAGIC ###### 1.0.2
# MAGIC 
# MAGIC **{"squared_error", "absolute_error", "poisson"}, default="squared_error"**
# MAGIC 
# MAGIC 
# MAGIC 用来衡量分枝质量的指标，即衡量不纯度的指标。
# MAGIC 
# MAGIC 支持：
# MAGIC * squared_error：均方误差，这等于方差约简作为特征选择标准
# MAGIC * absolute_error：平均绝对误差
# MAGIC * poisson：泊松偏差，使用泊松偏差的减少来查找分割
# MAGIC 
# MAGIC 使用"absolute_error"的训练比使用"squared_error"时要慢得多。
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=squared%5C%5Ferror%3D%5Cfrac%7B1%7D%7BN%7D%5Csum_%7Bi%3D1%7D%5EN%28f_i-y_i%29^2)
# MAGIC 
# MAGIC 其中![](https://www.zhihu.com/equation?tex=N)是样本数量，![](https://www.zhihu.com/equation?tex=i)是每一个数据样本，![](https://www.zhihu.com/equation?tex=f_i)是模型回归出的数值，![](https://www.zhihu.com/equation?tex=y_i)是样本点![](https://www.zhihu.com/equation?tex=i)实际的数值标签。所以![](https://www.zhihu.com/equation?tex=squared%5C%5Ferror)的本质，其实是样本真实数据与回归结果的差异。
# MAGIC 
# MAGIC **在回归树中，![](https://www.zhihu.com/equation?tex=squared%5C%5Ferror)不只是我们衡量分枝质量的指标，也是我们最常用的衡量回归树回归质量的指标。**
# MAGIC 
# MAGIC > 版本 0.18 新增：mae 平均绝对误差。  
# MAGIC 版本 0.24 新增：poisson 泊松偏差。
# MAGIC 
# MAGIC > 标准"mse"在 v1.0 中已弃用，并将在版本 1.2 中删除。使用等效的 criteria="squared_error"。  
# MAGIC 标准"mae"在 v1.0 中已弃用，并将在版本 1.2 中删除。使用等效的 criteria="absolute_error"。

# COMMAND ----------

# MAGIC %md
# MAGIC #### 随机性参数

# COMMAND ----------

# MAGIC %md
# MAGIC ##### random_state
# MAGIC 
# MAGIC **int, RandomState instance or None, default=None**
# MAGIC 
# MAGIC 控制构建树时所用样本的随机性（如果 bootstrap=True） 和在每个节点（如果 max_features < n_features）上查找最佳拆分时要考虑的特征的采样。
# MAGIC 
# MAGIC 随机森林的本质是一种袋装集成算法，袋装集成算法是对基评估器的预测结果取平均值原则或少数服从多数的原则来决定集成评估器的结果。随机森林判断错误的概率是：
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=E_%7Brandom%5C%5Fforest%7D%3D%5Csum_%7Bi%3De_%7Bmin%7D%7D%5ENC_N^iε^i%281-ε%29^%7BN-i%7D)
# MAGIC 
# MAGIC 其中，![](https://www.zhihu.com/equation?tex=N)是随机森林中树的数量，![](https://www.zhihu.com/equation?tex=i)是随机森林判断错误的树的数量，![](https://www.zhihu.com/equation?tex=e_%7Bmin%7D)是要判定随机森林判断错误至少需要满足的判断错误的树的数量，根据少数服从多数原则，通常![](https://www.zhihu.com/equation?tex=e_%7Bmin%7D>%5Cfrac%7BN%7D%7B2%7D)，![](https://www.zhihu.com/equation?tex=ε)是一棵树判断错误的概率，![](https://www.zhihu.com/equation?tex=%281-ε%29)是一棵树判断正确的概率，![](https://www.zhihu.com/equation?tex=%7BN-i%7D)是判断正确的次数。
# MAGIC 
# MAGIC 随机森林中的random_state用法与分类树中相似，只不过在分类树中，一个random_state只控制生成一棵树，而随机森林中的random_state控制的是生成森林的模式，而不是让一个森林中只有一棵树。
# MAGIC 
# MAGIC 对于固定的random_state生成的是同一片森林，森林中的每一棵树的random_state是不一样的，但由于生成的是同一片森林，所以多次训练产生的同一片森林中的每一棵树的random_state不会变。
# MAGIC 
# MAGIC 我们可以观察到，当random_state固定时，随机森林中生成的是固定的一片森林，也就是一组固定的树，但每棵树依然是不一样的，这是用“随机挑选特征进行分枝”的方法得到的随机性。并且我们可以证明，当这种随机性越大的时候，装袋法的效果一般会越来越好。**用装袋法集成时，基分类器应当时相互独立的，是不相同的**。
# MAGIC 
# MAGIC 但是这种做法的局限性是很强的，当我们需要成千上万棵树的时候，数据不一定能够提供成千上万的特征来让我们构筑尽量多的不同的树，因此，除了random_state，我们还需要其他的随机性。

# COMMAND ----------

# MAGIC %md
# MAGIC ##### bootstrap
# MAGIC 
# MAGIC **bool, default=True**
# MAGIC 
# MAGIC 构建树时是否使用引导样本（bootstrap samples）。如果为 False，则使用整个数据集来生成每个树。
# MAGIC 
# MAGIC 要让基分类器尽量都不一样，一种很容易理解的方法是使用不同的训练集来进行训练，而装袋法正是通过有放回的随机抽样技术来形成不同的训练数据，bootstrap就是用来控制抽样技术的参数。
# MAGIC 
# MAGIC 在一个含有n个样本的原始训练集中，我们进行随机采样，每次采样一个样本，并在抽取下一个样本之前将该样本放回到原始训练集，也就是说下次采样时这个样本依然可能被采集到，这样采集n次，最终得到一个和原始训练集一样大，n个样本组成的自助集。由于是随机采样，这样每次的自助集和原始数据集不同，和其他的采样集也是不同的。这样我们就可以自由创造取之不尽用之不竭、并且互不相同的自助集，用这些自助集来训练我们的基分类器，我们的基分类器自然也就各不相同了。
# MAGIC 
# MAGIC **bootstrap默认True，代表采用这种有放回的随机抽样技术。** 通常，这个参数不会被我们设置成False。
# MAGIC 
# MAGIC 然而有放回抽样也会有自己的问题。由于是有放回，一些样本可能在同一个自助集中出现多次，而其他一些却可能被忽略，一般来说，自助集大约平均会包含63%的原始数据。因为每一个样本被抽到某个自助集中的概率为：
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=1-%28+1-%5Cfrac%7B1%7D%7Bn%7D%29^n)
# MAGIC 
# MAGIC 当n足够大的时候，这个概率收敛于![](https://www.zhihu.com/equation?tex=1-%5Cfrac%7B1%7D%7Be%7D)，约等于0.632。因此，会有约37%的训练数据被浪费掉，没有参与建模，这些数据被称为袋外数据（out of bag data，简写为oob）。除了我们最开始就划分好的测试集之外，这些数据也可以被用来作为集成算法的测试集。**也就是说，在使用随机森林时，我们可以不划分测试集和训练集，只需要使用袋外数据来测试我们的模型即可**。当然，这也不是绝对的，当n和n_estimators都不够大的时候，很可能就没有数据掉落在袋外，自然也就无法使用oob数据来测试模型了。

# COMMAND ----------

# MAGIC %md
# MAGIC ##### oob_score
# MAGIC 
# MAGIC **bool, default=False**
# MAGIC 
# MAGIC 是否使用袋外样本来估计泛化精确性。仅当 bootstrap = True 时才可用。
# MAGIC 
# MAGIC 如果希望用袋外数据来测试，则需要在实例化时就将oob_score这个参数调整为True，训练完毕之后，我们可以用随机森林的一个属性 oob_score_ 来查看我们在袋外数据上测试的结果。

# COMMAND ----------

# MAGIC %md
# MAGIC #### 剪枝参数
# MAGIC 
# MAGIC 在不加限制的情况下，一棵决策树会生长到衡量不纯度的指标最优，或者没有更多的特征可用为止，这样的决策树往往会过拟合。
# MAGIC 
# MAGIC 为了让决策树有更好的泛化性，我们需要对决策树进行剪枝。

# COMMAND ----------

# MAGIC %md
# MAGIC ##### max_depth
# MAGIC 
# MAGIC **int, default=None**
# MAGIC 
# MAGIC 树的最大深度。如果为 None，则节点将展开，树辉持续生长，直到所有叶子都是纯的，或者直到所有叶子包含的样本少于 min_samples_split。

# COMMAND ----------

# MAGIC %md
# MAGIC ##### min_samples_split
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

# MAGIC %md
# MAGIC ##### min_samples_leaf
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

# MAGIC %md
# MAGIC ##### max_features
# MAGIC 
# MAGIC **int, float or {"auto", "sqrt", "log2"}, default="auto"**
# MAGIC 
# MAGIC 寻找最佳分枝时要考虑的特征数量：
# MAGIC 
# MAGIC * 如果为 int，则考虑每次分枝时的特征。max_features
# MAGIC * 如果是 float，则为比例，每次分枝时考虑的特征数目是max_featuresint(max_features * n_features)。
# MAGIC * 如果为"auto"，则 .max_features=n_features
# MAGIC * 如果为"sqrt"，则 .max_features=sqrt(n_features)
# MAGIC * 如果为"log2"，则 .max_features=log2(n_features)
# MAGIC * 如果为 None，则 .max_features=n_features
# MAGIC 
# MAGIC > 注意：如果在限制的 max_features 中，决策树无法找到节点样本上至少一个有效的分枝，那对分枝的搜索不会停止，决策树将会检查比限制的 max_features 数目更多的特征。
# MAGIC 
# MAGIC max_features 用来限制在分枝时考虑的特征个数，超过限制个数的特征都会被舍弃。
# MAGIC 
# MAGIC max_features 是用来限制高维度数据的过拟合的剪枝参数，但其方法比较暴力，是直接限制可以使用的特征的数量而强行使决策树停止生长。在不知道决策树中的各个特征的重要性的情况下，强行设定这个参数可能会导致模型学习不足。

# COMMAND ----------

# MAGIC %md
# MAGIC ##### max_leaf_nodes
# MAGIC 
# MAGIC **int, default=None**
# MAGIC 
# MAGIC 最大叶节点数量。在最佳分枝方式下，以max_leaf_nodes为限制来生长树。如果为 None，则叶节点数不受限制。

# COMMAND ----------

# MAGIC %md
# MAGIC ##### min_impurity_decrease
# MAGIC 
# MAGIC **float, default=0.0**
# MAGIC 
# MAGIC 如果节点的分裂导致不纯度的减少大于或等于此值，则节点将被分裂。
# MAGIC 
# MAGIC 加权不纯度减少方程如下：
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=%5Cfrac%7BN_t%7D%7BN%7D*%28+impurity-%5Cfrac%7BN_%7Bt_R%7D%7D%7BN_t%7D*right%5C%5F+impurity-%5Cfrac%7BN_%7Bt_L%7D%7D%7BN_t%7D*left%5C%5F+impurity%29)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=%5Cfrac%7BN_t%7D%7BN%7D*%28+不纯度-%5Cfrac%7BN_%7Bt_R%7D%7D%7BN_t%7D*右侧分枝的不纯度-%5Cfrac%7BN_%7Bt_L%7D%7D%7BN_t%7D*左侧分枝的不纯度%29)
# MAGIC 
# MAGIC 其中![](https://www.zhihu.com/equation?tex=N)是样本总数，![](https://www.zhihu.com/equation?tex=N_t)是当前节点上的样本数，![](https://www.zhihu.com/equation?tex=N_%7Bt_L%7D)是左子级中的样本数，![](https://www.zhihu.com/equation?tex=N_%7Bt_R%7D)是右子级中的样本数。
# MAGIC 
# MAGIC > 注意：如果 sample_weight 在 fit 接口中有值，则![](https://www.zhihu.com/equation?tex=N)、![](https://www.zhihu.com/equation?tex=N_t)、![](https://www.zhihu.com/equation?tex=N_%7Bt_L%7D)、![](https://www.zhihu.com/equation?tex=N_%7Bt_R%7D)都是指样本量的权重，而非单纯的样本数量。
# MAGIC 
# MAGIC > 0.19 版中的新功能。

# COMMAND ----------

# MAGIC %md
# MAGIC ##### min_impurity_split
# MAGIC 
# MAGIC **float, default=None**
# MAGIC 
# MAGIC 早期停止树木生长的阈值。如果节点的不纯度高于阈值，则节点将分裂，否则为叶子。
# MAGIC 
# MAGIC > 自版本 0.19 起弃用：min_impurity_split已弃用，转而支持 0.19 中的min_impurity_decrease。默认值 min_impurity_split 在 0.23 中已从 1e-7 更改为 0，**并将在 1.0 中删除（重命名为 0.25）。** 请改用min_impurity_decrease。

# COMMAND ----------

# MAGIC %md
# MAGIC #### 目标权重参数

# COMMAND ----------

# MAGIC %md
# MAGIC ##### min_weight_fraction_leaf
# MAGIC 
# MAGIC **float, default=0.0**
# MAGIC 
# MAGIC 一个叶节点要存在所需要的权重占输入模型的数据集的总权重的比例。
# MAGIC 
# MAGIC 总权重由fit接口中的sample_weight参数确定，当sample_weight为None时，默认所有样本的权重相同。

# COMMAND ----------

# MAGIC %md
# MAGIC #### n_jobs
# MAGIC 
# MAGIC **int, default=None**
# MAGIC 
# MAGIC 要并行运行的作业数。 fit, predict, decision_path 和 apply都在树上并行化。 None 表示 1，除非在joblib.parallel_backend上下文中。 -1 表示使用所有处理器。

# COMMAND ----------

# MAGIC %md
# MAGIC #### verbose
# MAGIC 
# MAGIC **int, default=0**
# MAGIC 
# MAGIC 控制拟合和预测时的详细程度。

# COMMAND ----------

# MAGIC %md
# MAGIC #### warm_start
# MAGIC 
# MAGIC **bool, default=False**
# MAGIC 
# MAGIC 设置为 True 时，重用上一个调用的解决方案以拟合并向融合中添加更多估计器，否则，只需拟合一个全新的林。

# COMMAND ----------

# MAGIC %md
# MAGIC #### ccp_alpha
# MAGIC 
# MAGIC **non-negative float, default=0.0**
# MAGIC 
# MAGIC 用于最小成本-复杂性修剪的复杂度参数。将选择的成本复杂度大于将选择的成本复杂度的子树。默认情况下，不执行修剪。
# MAGIC 
# MAGIC > 0.22 版中的新功能。

# COMMAND ----------

# MAGIC %md
# MAGIC #### max_samples
# MAGIC 
# MAGIC **int or float, default=None**
# MAGIC 
# MAGIC 如果 bootstrap 为 True，则表示要从 X 中提取以训练每个基本估计器的样本数。
# MAGIC 
# MAGIC * 如果为"None"（默认值），则绘制X.shape[0]个样本。
# MAGIC * 如果为 int，则绘制max_samples个样本。
# MAGIC * 如果是浮动，则绘制max_samples * X.shape[0]个样本。因此，max_samples应该在间隔(0, 1)内。
# MAGIC 
# MAGIC > 版本 0.22 中的新功能。

# COMMAND ----------

# MAGIC %md
# MAGIC ### 属性
# MAGIC 
# MAGIC 属性是在模型训练之后，能够调用查看的模型的各种性质。

# COMMAND ----------

# MAGIC %md
# MAGIC #### base_estimator_
# MAGIC 
# MAGIC 用于创建拟合子估计器集合的子估计器模板。

# COMMAND ----------

# MAGIC %md
# MAGIC #### estimators_
# MAGIC 
# MAGIC 拟合子估计器的集合。

# COMMAND ----------

# MAGIC %md
# MAGIC #### feature_importances_
# MAGIC 
# MAGIC 基于不纯度的特征重要性。

# COMMAND ----------

# MAGIC %md
# MAGIC #### n_features_
# MAGIC 
# MAGIC 执行fit时的特征数。

# COMMAND ----------

# MAGIC %md
# MAGIC #### n_outputs_
# MAGIC 
# MAGIC 执行fit时的输出数。

# COMMAND ----------

# MAGIC %md
# MAGIC #### oob_score_
# MAGIC 
# MAGIC 使用袋外估计获得的训练数据集的分数。仅当 oob_score 为 True 时，此属性才存在。

# COMMAND ----------

# MAGIC %md
# MAGIC #### oob_prediction_
# MAGIC 
# MAGIC 使用训练集上的袋外估计值计算预测。仅当 oob_score 为 True 时，此属性才存在。

# COMMAND ----------

# MAGIC %md
# MAGIC ### 方法
# MAGIC 
# MAGIC | 方法 | 描述 |
# MAGIC | ---- | ---- |
# MAGIC | apply(X) | 将森林中的树应用于 X，返回叶子索引。 |
# MAGIC | decision_path(X) | 返回森林中的决策路径。 |
# MAGIC | fit(X, y[, sample_weight]) | 从训练集 (X, y) 构建一片森林。|
# MAGIC | get_params([deep]) | 获取此估计器的参数。 |
# MAGIC | predict(X) | 预测 X 的回归目标。 |
# MAGIC | score(X, y[, sample_weight]) | 返回预测的确定系数![](https://www.zhihu.com/equation?tex=R^2)。 |
# MAGIC | set_params(**params) | 设置此估计器的参数。 |

# COMMAND ----------

# MAGIC %md
# MAGIC #### fit
# MAGIC 
# MAGIC **fit(X, y, sample_weight=None)**
# MAGIC 
# MAGIC 从训练集 (X, y) 构建一片森林。

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 参数
# MAGIC 
# MAGIC * **X**: {array-like, sparse matrix} of shape (n_samples, n_features)  
# MAGIC 训练输入示例。在内部，将会转换为dtype=np.float32，如果提供给稀疏矩阵，则将其转换为csc_matrix。
# MAGIC * **y**: array-like of shape (n_samples,) or (n_samples, n_outputs)  
# MAGIC 目标值（分类中的类标签、回归中的实数）。
# MAGIC * **sample_weight**: array-like of shape (n_samples,), default=None  
# MAGIC 样本权重。如果为 None，则样本的权重相等。在每个节点中搜索拆分时，将忽略将创建具有净零或负权重的子节点的拆分。如果拆分会导致任何单个类在任一子节点中承载负权重，则也会忽略拆分。

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 返回
# MAGIC 
# MAGIC * **self**: RandomForestRegressor  
# MAGIC 拟合估计器。

# COMMAND ----------

# MAGIC %md
# MAGIC #### score
# MAGIC 
# MAGIC **score(X, y, sample_weight=None)**
# MAGIC 
# MAGIC 返回预测的确定系数![](https://www.zhihu.com/equation?tex=R^2)。
# MAGIC 
# MAGIC 系数![](https://www.zhihu.com/equation?tex=R^2)定义为![](https://www.zhihu.com/equation?tex=%28+1-%5Cfrac%7Bu%7D%7Bv%7D%29)，其中![](https://www.zhihu.com/equation?tex=u)是残差平方和![](https://www.zhihu.com/equation?tex=%5Csum_%7Bi%3D1%7D%5EN%28f_i-y_i%29^2)，![](https://www.zhihu.com/equation?tex=v)是平方的总和![](https://www.zhihu.com/equation?tex=%5Csum_%7Bi%3D1%7D%5EN%28y_i-%5Chat%7By%7D%29^2)。最佳可能分数为 1.0，并且可能为负数（因为模型可能任意更差）。始终预测![](https://www.zhihu.com/equation?tex=y)的预期值的常量模型（不考虑输入特征）![](https://www.zhihu.com/equation?tex=R^2)将获得 0.0 的分数。
# MAGIC 
# MAGIC **回归树的score返回的是![](https://www.zhihu.com/equation?tex=R^2)，并不是![](https://www.zhihu.com/equation?tex=MSE)。**
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=R^2%3D1-%5Cfrac%7Bu%7D%7Bv%7D)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=u%3D%5Csum_%7Bi%3D1%7D%5EN%28f_i-y_i%29^2)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=v%3D%5Csum_%7Bi%3D1%7D%5EN%28y_i-%5Chat%7By%7D%29^2)
# MAGIC 
# MAGIC 其中![](https://www.zhihu.com/equation?tex=N)是样本数量，![](https://www.zhihu.com/equation?tex=i)是每一个数据样本，![](https://www.zhihu.com/equation?tex=f_i)是模型回归出的数值，![](https://www.zhihu.com/equation?tex=y_i)是样本点![](https://www.zhihu.com/equation?tex=i)实际的数值标签，![](https://www.zhihu.com/equation?tex=%5Chat%7By%7D)是真实数值标签的平均数，![](https://www.zhihu.com/equation?tex=R^2)可以为正、为负，而均方差永远为正。

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 参数
# MAGIC 
# MAGIC * **X**: array-like of shape (n_samples, n_features)  
# MAGIC 测试样本。对于一些估计器，这可能是一个预先计算的核矩阵或泛型对象的列表，而不是形状(n_samples, n_samples_fitted)，其中n_samples_fitted是估计器的拟合中使用的样本数。
# MAGIC * **y**: array-like of shape (n_samples,) or (n_samples, n_outputs)  
# MAGIC X的真实标签。
# MAGIC * **sample_weight**: array-like of shape (n_samples,), default=None  
# MAGIC 样本权重。

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 返回
# MAGIC 
# MAGIC * **score**: float  
# MAGIC ![](https://www.zhihu.com/equation?tex=R^2)

# COMMAND ----------

# MAGIC %md
# MAGIC #### apply
# MAGIC 
# MAGIC **apply(X)**
# MAGIC 
# MAGIC 返回预测每个样本的叶子的索引。

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 参数
# MAGIC 
# MAGIC * **X**: {array-like, sparse matrix} of shape (n_samples, n_features)  
# MAGIC 训练输入示例。在内部，将会转换为dtype=np.float32，如果提供给稀疏矩阵，则将其转换为csr_matrix。

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 返回
# MAGIC 
# MAGIC * **X_leaves**: ndarray of shape (n_samples, n_estimators)
# MAGIC 对于 X 中的每个数据点 x 和森林中的每棵树，返回叶子 x 的索引。

# COMMAND ----------

# MAGIC %md
# MAGIC #### predict
# MAGIC 
# MAGIC **predict(X)**
# MAGIC 
# MAGIC 预测 X 的回归目标。
# MAGIC 
# MAGIC 输入样本的预测回归目标计算为森林中树的平均预测回归目标。

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 参数
# MAGIC 
# MAGIC * **X**: {array-like, sparse matrix} of shape (n_samples, n_features)  
# MAGIC 训练输入示例。在内部，将会转换为dtype=np.float32，如果提供给稀疏矩阵，则将其转换为csr_matrix。

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 返回
# MAGIC 
# MAGIC * **y**: ndarray of shape (n_samples,) or (n_samples, n_outputs)  
# MAGIC 预测回归目标。
