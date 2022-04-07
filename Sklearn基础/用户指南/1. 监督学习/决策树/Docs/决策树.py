# Databricks notebook source
# MAGIC %md
# MAGIC # 决策树

# COMMAND ----------

# MAGIC %md
# MAGIC ## 概述

# COMMAND ----------

# MAGIC %md
# MAGIC ### 决策树是怎样工作的
# MAGIC 
# MAGIC **决策树(Decision Tree)** 是一种用于分类和回归的非参数监督学习方法。目标是创建一个模型，通过学习从数据特征推断的简单决策规则来预测目标变量的值，并用树状图的结构来呈现这些规则。树可以看作是一个分段常数近似。
# MAGIC 
# MAGIC 决策的过程，一直是在对记录的特征进行推断。最初的特征推断所在的地方叫做**根节点**，在得到结论前的每一个特征的推断都是**中间节点**，而得到的每一个结论都叫做**叶子节点**。
# MAGIC 
# MAGIC > 关键概念：**节点**
# MAGIC >
# MAGIC > - 根节点：最初的特征的推断
# MAGIC > - 中间节点：得到结论前的特征的推断
# MAGIC > - 叶子节点：特征推断得到的结论，**每个叶子节点都是一个类别标签**
# MAGIC 
# MAGIC **决策树的一些优点是：**
# MAGIC 
# MAGIC * 易于理解和解释。树可以被可视化。
# MAGIC * 几乎不需要数据准备。其他技术通常需要数据规范化，需要创建虚拟变量并删除空白值。但请注意，此模块不支持缺失值。
# MAGIC * 使用树（即预测数据）的成本与用于训练树的数据点数是对数。
# MAGIC * 能够处理数字和分类数据。但是，scikit-learn 实现目前不支持分类变量。其他技术通常专门用于分析只有一种类型的变量的数据集。
# MAGIC * 能够处理多输出问题。
# MAGIC * 使用白盒模型。如果给定的情况在模型中是可观察的，则对条件的解释很容易用布尔逻辑来解释。相比之下，在黑盒模型中（例如，在人工神经网络中），结果可能更难以解释。
# MAGIC * 可以使用统计测试来验证模型。这样就可以考虑模型的可靠性。
# MAGIC * 即使其假设与生成数据的真实模型有些冲突，性能也很好。
# MAGIC 
# MAGIC **决策树的一些缺点是：**
# MAGIC 
# MAGIC * 决策树学习者可以创建过于复杂的树，这些树不能很好地概括数据。这称为**过拟合**。为了避免此问题，需要修剪，设置叶节点所需的最小样本数或设置树的最大深度等机制。
# MAGIC * 决策树可能不稳定，因为数据中的微小变化可能会导致生成完全不同的树。通过在集成中使用决策树可以缓解此问题。
# MAGIC * 决策树的预测既不是平滑的也不是连续的，而是分段的常数近似。因此，他们不擅长外推。
# MAGIC * 众所周知，在最优性的几个方面，甚至对于简单的概念，学习最优决策树的问题是NP完备的。因此，实用的决策树学习算法基于启发式算法，例如贪婪算法，其中在每个节点上做出局部最优决策。此类算法无法保证返回全局最优决策树。这可以通过在集成学习器中训练多个树来缓解，其中特征和样本通过替换随机采样。
# MAGIC * 有些概念很难学习，因为决策树不容易表达它们，例如异或奇偶校验或多路复用器问题。
# MAGIC * 如果某些类占主导地位，则决策树学习器会创建有偏差的树。因此，建议在与决策树拟合之前平衡数据集。
# MAGIC 
# MAGIC **决策树算法的核心是要解决两个问题：**
# MAGIC 
# MAGIC - 如何从数据表中找出最佳节点和最佳分枝
# MAGIC - 如何让决策树停止生长，防止过拟合

# COMMAND ----------

# MAGIC %md
# MAGIC ## sklearn中的决策树

# COMMAND ----------

# MAGIC %md
# MAGIC ### 模块 sklearn.tree
# MAGIC 
# MAGIC sklearn中决策树的类都在“tree”这个模块下。
# MAGIC 
# MAGIC | 类                          | 描述                                  |
# MAGIC | --------------------------- | ------------------------------------- |
# MAGIC | tree.DecisionTreeClassifier | 分类树                                |
# MAGIC | tree.DecisionTreeRegressor  | 回归树                                |
# MAGIC | tree.export_graphviz        | 将生成的决策树导出为DOT格式，画图专用 |
# MAGIC | tree.ExtraTreeClassifier    | 高随机版本的分类树                    |
# MAGIC | tree.ExtraTreeRegressor     | 高随机版本的回归树                    |

# COMMAND ----------

# MAGIC %md
# MAGIC ### sklearn建模流程
# MAGIC 
# MAGIC 1、实例化，建立评估模型对象
# MAGIC 
# MAGIC 2、通过模型接口训练模型
# MAGIC 
# MAGIC 3、通过模型接口提取需要的信息

# COMMAND ----------

from sklearn import tree               # 导入需要的模块

clf = tree.DecisionTreeClassifier()    # 实例化
clf = clf.fit(x_train, y_train)        # 用训练集数据训练模型
result = clf.score(x_test, y_test)     # 导入测试集，从接口中调用需要的信息

# COMMAND ----------

# MAGIC %md
# MAGIC ## DecisionTreeClassifier
# MAGIC 
# MAGIC DecisionTreeClassifier 是一个能够对数据集执行多类分类的类。
# MAGIC 
# MAGIC DecisionTreeClassifier 能够进行二元分类（其中标签为 [-1， 1]）和多类（其中标签为 [0， ...， K-1]）分类。
# MAGIC 
# MAGIC * 0.24
# MAGIC > class sklearn.tree.DecisionTreeClassifier(*, criterion='gini', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0, **min_impurity_split=None,** class_weight=None, ccp_alpha=0.0)
# MAGIC 
# MAGIC * 1.0.2
# MAGIC > class sklearn.tree.DecisionTreeClassifier(*, criterion='gini', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0, class_weight=None, ccp_alpha=0.0)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 参数

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
# MAGIC 
# MAGIC > 熵：熵表示随机变量不确定性的度量  
# MAGIC 信息增益：表示特征X使得类Y的不确定性减少的程度

# COMMAND ----------

# MAGIC %md
# MAGIC #### 随机性参数
# MAGIC 
# MAGIC score会在某个值附近波动，每次画出来的树不一样。它为什么会不稳定？
# MAGIC 
# MAGIC 无论决策树模型如何进化，他在分枝上的本质都还是追求某个不纯度相关的指标的优化。不纯度是基于节点来计算的，也就是说，决策树在建树时，是靠优化节点来追求一棵优化的树的。sklearn表示，既然一棵树不能保证最优，那就建更多的不同的树，然后从中选取最好的。在每次分枝时，不使用全部特征，而是随机选取一部分特征，从中选取不纯度相关指标最优的作为分枝用的节点。这样，每次生成的树也就不同了。

# COMMAND ----------

# MAGIC %md
# MAGIC ##### random_state
# MAGIC 
# MAGIC **int, RandomState instance or None, default=None**
# MAGIC 
# MAGIC 用来设置分枝中的随机模式的参数，控制随机数生成器的随机性。每次分枝时，这些要素始终是随机排列的。
# MAGIC 
# MAGIC 在高维度时随机性会表现更明显，低维度的数据（比如鸢尾花数据集），随机性几乎不会显现。设置任意整数，会一直生长出同一棵树，让模型稳定下来。
# MAGIC 
# MAGIC * 输入 int，则 random_state 是由随机数生成器生成的随机数种子
# MAGIC * 输入 RandomState instance，则 random_state 是一个随机数生成器
# MAGIC * 输入 None，随机数生成器会是 np.random 模块中的一个RandomState实例

# COMMAND ----------

# MAGIC %md
# MAGIC ##### splitter
# MAGIC 
# MAGIC **{"best", "random"}, default="best"**
# MAGIC 
# MAGIC 用于在每个节点上选择分枝的策略。
# MAGIC 
# MAGIC 用来控制决策树中的随机选项的。
# MAGIC 
# MAGIC * best：选择最佳分枝
# MAGIC * random：选择最佳随机分枝，树会更深，对训练集的拟合将会降低

# COMMAND ----------

# MAGIC %md
# MAGIC #### 剪枝参数
# MAGIC 
# MAGIC 在不加限制的情况下，一棵决策树会生长到衡量不纯度的指标最优，或者没有更多的特征可用为止，这样的决策树往往会过拟合。
# MAGIC 
# MAGIC 为了让决策树有更好的泛化性，我们需要对决策树进行剪枝。
# MAGIC 
# MAGIC **剪枝策略对决策树的影响巨大，正确的剪枝策略是优化决策树算法的核心。**

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
# MAGIC **int, float or {"auto", "sqrt", "log2"}, default=None**
# MAGIC 
# MAGIC 寻找最佳分枝时要考虑的特征数量：
# MAGIC 
# MAGIC * 如果为 int，则考虑每次分枝时的特征。max_features
# MAGIC * 如果是 float，则为比例，每次分枝时考虑的特征数目是max_featuresint(max_features * n_features)。
# MAGIC * 如果为"auto"，则 .max_features=sqrt(n_features)
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
# MAGIC **float, default=0**
# MAGIC 
# MAGIC 早期停止树木生长的阈值。如果节点的不纯度高于阈值，则节点将分裂，否则为叶子。
# MAGIC 
# MAGIC > 自版本 0.19 起弃用：min_impurity_split已弃用，转而支持 0.19 中的min_impurity_decrease。默认值 min_impurity_split 在 0.23 中已从 1e-7 更改为 0，**并将在 1.0 中删除（重命名为 0.25）。** 请改用min_impurity_decrease。

# COMMAND ----------

# MAGIC %md
# MAGIC #### 目标权重参数
# MAGIC 
# MAGIC 完成样本标签平衡的参数，样本不平衡是指在一组数据集中，有一类标签天生占有很大的比例。比如说，在银行要判断“一个办了信用卡的人是否会违约”，不会违约的人可能会占99%。这种分类情况下，即便模型什么也不做，直接将所有结果预测成不会违约，那正确率也能有99%。因此，我们需要使用class_weight参数对样本标签进行一定的均衡，给少量的标签更多的权重，让模型更偏向少数类，向捕获少数类的方向建模。
# MAGIC 
# MAGIC 有了权重之后，样本量就不再是单纯地记录数目，而是受输入的权重影响了，因此这时候剪枝，就需要搭配min_weight_fraction_leaf这个基于权重的剪枝参数来使用。
# MAGIC 
# MAGIC > 基于权重的剪枝参数将比不知道样本权重的标准剪枝参数更少偏向主导类。如果样本是加权的则使用基于权重的预修剪标准更容易优化树结构，着确保叶节点至少包含样本权重的总和的一小部分。

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
# MAGIC **dict, list of dict or “balanced”, default=None**
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
# MAGIC #### ccp_alpha
# MAGIC 
# MAGIC **non-negative float, default=0.0**
# MAGIC 
# MAGIC 用于最小成本-复杂性修剪的复杂度参数。将选择的成本复杂度大于将选择的成本复杂度的子树。默认情况下，不执行修剪。
# MAGIC 
# MAGIC > 0.22 版中的新功能。

# COMMAND ----------

# MAGIC %md
# MAGIC ### 属性
# MAGIC 
# MAGIC 属性是在模型训练之后，能够调用查看的模型的各种性质。
# MAGIC 
# MAGIC 对决策树来说，最重要的是feature_importances_，它能够查看各个特征对模型的重要性。

# COMMAND ----------

# MAGIC %md
# MAGIC #### classes_
# MAGIC 
# MAGIC 类标签（单输出问题）或类标签数组列表（多输出问题）。

# COMMAND ----------

# MAGIC %md
# MAGIC #### feature_importances_
# MAGIC 
# MAGIC 返回特征的重要性。

# COMMAND ----------

# MAGIC %md
# MAGIC #### max_features_
# MAGIC 
# MAGIC max_features的推断值。

# COMMAND ----------

# MAGIC %md
# MAGIC #### n_classes_
# MAGIC 
# MAGIC 类数（对于单个输出问题），或包含每个输出的类数的列表（对于多输出问题）。

# COMMAND ----------

# MAGIC %md
# MAGIC #### n_features_
# MAGIC 
# MAGIC fit执行时的特征数。
# MAGIC 
# MAGIC > 自版本 1.0 起已弃用，在版本 1.2 中将会被移除。

# COMMAND ----------

# MAGIC %md
# MAGIC #### n_features_in_
# MAGIC 
# MAGIC 在拟合过程中看到的特征数量。
# MAGIC 
# MAGIC > 0.24 版中新增。

# COMMAND ----------

# MAGIC %md
# MAGIC #### n_outputs_
# MAGIC 
# MAGIC fit执行时的输出数。

# COMMAND ----------

# MAGIC %md
# MAGIC #### tree_
# MAGIC 
# MAGIC 基础树对象。
# MAGIC 
# MAGIC 输出一个可以导出建好的树结构的端口，通过这个端口，可以访问树的结构和低级属性，包括但不限于查看：
# MAGIC * 二叉树的结构
# MAGIC * 每个节点的深度以及它是否是叶子
# MAGIC * 使用decision_path方法的示例到达的节点
# MAGIC * 用apply这个接口取样出的叶子

# COMMAND ----------

# MAGIC %md
# MAGIC #### feature_names_in_
# MAGIC 
# MAGIC 拟合期间看到的特征的名称。仅当X的要素名称均为字符串时定义。
# MAGIC 
# MAGIC > 1.0 版中新增。

# COMMAND ----------

# MAGIC %md
# MAGIC ### 方法
# MAGIC 
# MAGIC | 方法 | 描述 |
# MAGIC | ---- | ---- |
# MAGIC | apply(X[, check_input]) | 返回预测每个样本的叶子的索引。 |
# MAGIC | cost_complexity_pruning_path(X, y[, …]) | 在最小成本复杂性修剪期间计算修剪路径。 |
# MAGIC | decision_path(X[, check_input]) | 返回树中的决策路径。 |
# MAGIC | fit(X, y[, sample_weight, check_input, …]) | 从训练集 (X, y) 生成决策树分类器。 |
# MAGIC | get_depth() | 返回决策树的深度。 |
# MAGIC | get_n_leaves() | 返回决策树的叶子数。 |
# MAGIC | get_params([deep]) | 获取此估计器的参数。 |
# MAGIC | predict(X[, check_input]) | 预测 X 的类或回归值。 |
# MAGIC | predict_log_proba(X) | 预测输入样本 X 的类对数概率。 |
# MAGIC | predict_proba(X[, check_input]) | 预测输入样本 X 的类概率。 |
# MAGIC | score(X, y[, sample_weight]) | 返回给定测试数据和标签的平均精度。 |
# MAGIC | set_params(**params) | 设置此估计器的参数。 |

# COMMAND ----------

# MAGIC %md
# MAGIC #### fit
# MAGIC 
# MAGIC **fit(X, y[, sample_weight, check_input, …])**
# MAGIC 
# MAGIC 从训练集 (X, y) 生成决策树分类器。

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 参数
# MAGIC 
# MAGIC * **X**: {array-like, sparse matrix} of shape (n_samples, n_features)  
# MAGIC 训练输入示例。在内部，将会转换为dtype=np.float32，如果提供给稀疏矩阵，则将其转换为csc_matrix。
# MAGIC * **y**: array-like of shape (n_samples,) or (n_samples, n_outputs)  
# MAGIC 目标值（类标签）为整数或字符串。
# MAGIC * **sample_weight**: array-like of shape (n_samples,), default=None  
# MAGIC 样本权重。如果为 None，则样本的权重相等。在每个节点中搜索拆分时，将忽略将创建具有净零或负权重的子节点的拆分。如果拆分会导致任何单个类在任一子节点中承载负权重，则也会忽略拆分。
# MAGIC * **check_input**: bool, default=True  
# MAGIC 允许绕过多个输入检查。除非您知道自己做了什么，否则不要使用此参数。
# MAGIC * **X_idx_sorted**: deprecated, default="deprecated"  
# MAGIC 此参数已弃用，不起作用。它将在 1.1 中删除（重命名为 0.26）。
# MAGIC > 自版本 0.24 起已弃用。

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 返回
# MAGIC 
# MAGIC * **self**: DecisionTreeClassifier  
# MAGIC 拟合估计器。

# COMMAND ----------

# MAGIC %md
# MAGIC #### score
# MAGIC 
# MAGIC **score(X, y, sample_weight=None)**
# MAGIC 
# MAGIC 返回给定测试数据和标签的平均精度。

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
# MAGIC **apply(X, check_input=True)**
# MAGIC 
# MAGIC 返回预测每个样本的叶子的索引。
# MAGIC 
# MAGIC 版本 0.17 中的新功能。

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 参数
# MAGIC 
# MAGIC * **X**: {array-like, sparse matrix} of shape (n_samples, n_features)  
# MAGIC 训练输入示例。在内部，将会转换为dtype=np.float32，如果提供给稀疏矩阵，则将其转换为csr_matrix。
# MAGIC * **check_input**: bool, default=True  
# MAGIC 允许绕过多个输入检查。除非您知道自己做了什么，否则不要使用此参数。

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 返回
# MAGIC 
# MAGIC * **X_leaves**: array-like of shape (n_samples,)  
# MAGIC 对于 X 中的每个数据点 x，返回叶子 x 的索引。叶子的编号在[0, self.tree_.node_count)内，可能在编号上有间隙。

# COMMAND ----------

# MAGIC %md
# MAGIC #### predict
# MAGIC 
# MAGIC **predict(X, check_input=True)**
# MAGIC 
# MAGIC 预测 X 的类或回归值。
# MAGIC 
# MAGIC 对于分类模型，将返回 X 中每个样本的预测类。对于回归模型，将返回基于 X 的预测值。

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 参数
# MAGIC 
# MAGIC * **X**: {array-like, sparse matrix} of shape (n_samples, n_features)  
# MAGIC 训练输入示例。在内部，将会转换为dtype=np.float32，如果提供给稀疏矩阵，则将其转换为csr_matrix。
# MAGIC * **check_input**: bool, default=True  
# MAGIC 允许绕过多个输入检查。除非您知道自己做了什么，否则不要使用此参数。

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 返回
# MAGIC 
# MAGIC * **y**: array-like of shape (n_samples,) or (n_samples, n_outputs)  
# MAGIC 预测类或预测值。

# COMMAND ----------

# MAGIC %md
# MAGIC ## DecisionTreeRegressor
# MAGIC 
# MAGIC 决策树回归器。
# MAGIC 
# MAGIC * 0.24
# MAGIC > class sklearn.tree.DecisionTreeRegressor(*, criterion='mse', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0, **min_impurity_split=None,** ccp_alpha=0.0)
# MAGIC 
# MAGIC * 1.0.2
# MAGIC > class sklearn.tree.DecisionTreeRegressor(*, criterion='squared_error', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0, ccp_alpha=0.0)
# MAGIC 
# MAGIC 几乎所有参数，属性和接口都和分类树一样。
# MAGIC 
# MAGIC 需要注意的是，在回归树中，没有标签分布是否均衡的问题。

# COMMAND ----------

# MAGIC %md
# MAGIC ### 参数

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
# MAGIC **{"mse", "friedman_mse", "mae", "poisson"}, default="mse"**
# MAGIC 
# MAGIC 用来衡量分枝质量的指标，即衡量不纯度的指标。
# MAGIC 
# MAGIC 支持：
# MAGIC * mse：均方误差，父节点和叶子节点之间的均方误差的差额将被用来作为特征选择的标准，这种方法通过使用叶子节点的均值来最小化L2损失
# MAGIC * friedman_mse：费尔德曼均方误差，这种指标使用费尔德曼针对潜在分枝中的问题改进后的均方误差
# MAGIC * mae：平均绝对误差，它使用叶节点的中值来最小化 L1 损失
# MAGIC * poisson：泊松偏差
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=MSE%3D%5Cfrac%7B1%7D%7BN%7D%5Csum_%7Bi%3D1%7D%5EN%28f_i-y_i%29^2)
# MAGIC 
# MAGIC 其中![](https://www.zhihu.com/equation?tex=N)是样本数量，![](https://www.zhihu.com/equation?tex=i)是每一个数据样本，![](https://www.zhihu.com/equation?tex=f_i)是模型回归出的数值，![](https://www.zhihu.com/equation?tex=y_i)是样本点![](https://www.zhihu.com/equation?tex=i)实际的数值标签。所以![](https://www.zhihu.com/equation?tex=MSE)的本质，其实是样本真实数据与回归结果的差异。
# MAGIC 
# MAGIC **在回归树中，![](https://www.zhihu.com/equation?tex=MSE)不只是我们衡量分枝质量的指标，也是我们最常用的衡量回归树回归质量的指标。**
# MAGIC 
# MAGIC > 版本 0.18 新增：mae 平均绝对误差。  
# MAGIC 版本 0.24 新增：poisson 泊松偏差。

# COMMAND ----------

# MAGIC %md
# MAGIC ###### 1.0.2
# MAGIC 
# MAGIC **{"squared_error", "friedman_mse", "absolute_error", "poisson"}, default="squared_error"**
# MAGIC 
# MAGIC 用来衡量分枝质量的指标，即衡量不纯度的指标。
# MAGIC 
# MAGIC 支持：
# MAGIC * squared_error：均方误差，父节点和叶子节点之间的均方误差的差额将被用来作为特征选择的标准，这种方法通过使用叶子节点的均值来最小化L2损失
# MAGIC * friedman_mse：费尔德曼均方误差，这种指标使用费尔德曼针对潜在分枝中的问题改进后的均方误差
# MAGIC * absolute_error：平均绝对误差，它使用叶节点的中值来最小化 L1 损失
# MAGIC * poisson：泊松偏差
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
# MAGIC 用来设置分枝中的随机模式的参数，控制随机数生成器的随机性。每次分枝时，这些要素始终是随机排列的。
# MAGIC 
# MAGIC 在高维度时随机性会表现更明显，低维度的数据（比如鸢尾花数据集），随机性几乎不会显现。设置任意整数，会一直生长出同一棵树，让模型稳定下来。
# MAGIC 
# MAGIC * 输入 int，则 random_state 是由随机数生成器生成的随机数种子
# MAGIC * 输入 RandomState instance，则 random_state 是一个随机数生成器
# MAGIC * 输入 None，随机数生成器会是 np.random 模块中的一个RandomState实例

# COMMAND ----------

# MAGIC %md
# MAGIC ##### splitter
# MAGIC 
# MAGIC **{"best", "random"}, default="best"**
# MAGIC 
# MAGIC 用于在每个节点上选择分枝的策略。
# MAGIC 
# MAGIC 用来控制决策树中的随机选项的。
# MAGIC 
# MAGIC * best：选择最佳分枝
# MAGIC * random：选择最佳随机分枝，树会更深，对训练集的拟合将会降低

# COMMAND ----------

# MAGIC %md
# MAGIC #### 剪枝参数

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
# MAGIC **int, float or {"auto", "sqrt", "log2"}, default=None**
# MAGIC 
# MAGIC 寻找最佳分枝时要考虑的特征数量：
# MAGIC 
# MAGIC * 如果为 int，则考虑每次分枝时的特征。max_features
# MAGIC * 如果是 float，则为比例，每次分枝时考虑的特征数目是max_featuresint(max_features * n_features)。
# MAGIC * 如果为"auto"，则 .max_features=**n_features**
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
# MAGIC **float, default=0**
# MAGIC 
# MAGIC 早期停止树木生长的阈值。如果节点的不纯度高于阈值，则节点将分裂，否则为叶子。
# MAGIC 
# MAGIC > 自版本 0.19 起弃用：min_impurity_split已弃用，转而支持 0.19 中的min_impurity_decrease。默认值 min_impurity_split 在 0.23 中已从 1e-7 更改为 0，**并将在 1.0 中删除（重命名为 0.25）。** 请改用min_impurity_decrease。

# COMMAND ----------

# MAGIC %md
# MAGIC ##### max_leaf_nodes
# MAGIC 
# MAGIC **int, default=None**
# MAGIC 
# MAGIC 最大叶节点数量。在最佳分枝方式下，以max_leaf_nodes为限制来生长树。如果为 None，则叶节点数不受限制。

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
# MAGIC #### ccp_alpha
# MAGIC 
# MAGIC **non-negative float, default=0.0**
# MAGIC 
# MAGIC 用于最小成本-复杂性修剪的复杂度参数。将选择的成本复杂度大于将选择的成本复杂度的子树。默认情况下，不执行修剪。
# MAGIC 
# MAGIC > 0.22 版中的新功能。

# COMMAND ----------

# MAGIC %md
# MAGIC ### 属性

# COMMAND ----------

# MAGIC %md
# MAGIC #### feature_importances_
# MAGIC 
# MAGIC 返回特征的重要性。

# COMMAND ----------

# MAGIC %md
# MAGIC #### max_features_
# MAGIC 
# MAGIC max_features的推断值。

# COMMAND ----------

# MAGIC %md
# MAGIC #### n_features_
# MAGIC 
# MAGIC fit执行时的特征数。
# MAGIC 
# MAGIC > 自版本 1.0 起已弃用，在版本 1.2 中将会被移除。

# COMMAND ----------

# MAGIC %md
# MAGIC #### n_features_in_
# MAGIC 
# MAGIC 在拟合过程中看到的特征数量。
# MAGIC 
# MAGIC > 0.24 版中新增。

# COMMAND ----------

# MAGIC %md
# MAGIC #### n_outputs_
# MAGIC 
# MAGIC fit执行时的输出数。

# COMMAND ----------

# MAGIC %md
# MAGIC #### tree_
# MAGIC 
# MAGIC 基础树对象。
# MAGIC 
# MAGIC 输出一个可以导出建好的树结构的端口，通过这个端口，可以访问树的结构和低级属性，包括但不限于查看：
# MAGIC * 二叉树的结构
# MAGIC * 每个节点的深度以及它是否是叶子
# MAGIC * 使用decision_path方法的示例到达的节点
# MAGIC * 用apply这个接口取样出的叶子

# COMMAND ----------

# MAGIC %md
# MAGIC #### feature_names_in_
# MAGIC 
# MAGIC 拟合期间看到的特征的名称。仅当X的要素名称均为字符串时定义。
# MAGIC 
# MAGIC > 1.0 版中新增。

# COMMAND ----------

# MAGIC %md
# MAGIC ### 方法
# MAGIC 
# MAGIC | 方法 | 描述 |
# MAGIC | ---- | ---- |
# MAGIC | apply(X[, check_input]) | 返回预测每个样本的叶子的索引。 |
# MAGIC | cost_complexity_pruning_path(X, y[, …]) | 在最小成本复杂性修剪期间计算修剪路径。 |
# MAGIC | decision_path(X[, check_input]) | 返回树中的决策路径。 |
# MAGIC | fit(X, y[, sample_weight, check_input, …]) | 从训练集 (X, y) 生成决策树回归器。 |
# MAGIC | get_depth() | 返回决策树的深度。 |
# MAGIC | get_n_leaves() | 返回决策树的叶子数。 |
# MAGIC | get_params([deep]) | 获取此估计器的参数。 |
# MAGIC | predict(X[, check_input]) | 预测 X 的类或回归值。 |
# MAGIC | predict_log_proba(X) | 预测输入样本 X 的类对数概率。 |
# MAGIC | predict_proba(X[, check_input]) | 预测输入样本 X 的类概率。 |
# MAGIC | score(X, y[, sample_weight]) | 返回给定测试数据和标签的平均精度。 |
# MAGIC | set_params(**params) | 设置此估计器的参数。 |

# COMMAND ----------

# MAGIC %md
# MAGIC #### fit
# MAGIC 
# MAGIC **fit(X, y[, sample_weight, check_input, …])**
# MAGIC 
# MAGIC 从训练集 (X, y) 生成决策树回归器。

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 参数
# MAGIC 
# MAGIC * **X**: {array-like, sparse matrix} of shape (n_samples, n_features)  
# MAGIC 训练输入示例。在内部，将会转换为dtype=np.float32，如果提供给稀疏矩阵，则将其转换为csc_matrix。
# MAGIC * **y**: array-like of shape (n_samples,) or (n_samples, n_outputs)  
# MAGIC 目标值（实数），使用 dtype=np.float64 和 order='C' 可实现最高效率。
# MAGIC * **sample_weight**: array-like of shape (n_samples,), default=None  
# MAGIC 样本权重。如果为 None，则样本的权重相等。在每个节点中搜索拆分时，将忽略将创建具有净零或负权重的子节点的拆分。
# MAGIC * **check_input**: bool, default=True  
# MAGIC 允许绕过多个输入检查。除非您知道自己做了什么，否则不要使用此参数。
# MAGIC * **X_idx_sorted**: deprecated, default="deprecated"  
# MAGIC 此参数已弃用，不起作用。它将在 1.1 中删除（重命名为 0.26）。
# MAGIC > 自版本 0.24 起已弃用。

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 返回
# MAGIC 
# MAGIC * **self**: DecisionTreeRegressor  
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
# MAGIC 
# MAGIC 值得一提的是，**虽然均方差永远为正，但sklearn当中使用均方误差作为评判标准时，却是计算“负均方误差”（neg_mean_squared_error）**。这是因为sklearn在计算模型评估指标的时候，会考虑指标本身的性质，均方误差本身是一种误差，所以被sklearn划分为模型的一种损失，因此在sklearn中都是以负数表示。

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 参数
# MAGIC 
# MAGIC * **X**: array-like of shape (n_samples, n_features)  
# MAGIC 测试样本。对于某些估计器，这可能是一个预计算的核矩阵或一个通用对象列表，而不是形状(n_samples, n_samples_fitted)，其中n_samples_fitted是估计器拟合中使用的样本数。
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
# MAGIC **apply(X, check_input=True)**
# MAGIC 
# MAGIC 返回预测每个样本的叶子的索引。
# MAGIC 
# MAGIC 版本 0.17 中的新功能。

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 参数
# MAGIC 
# MAGIC * **X**: {array-like, sparse matrix} of shape (n_samples, n_features)  
# MAGIC 训练输入示例。在内部，将会转换为dtype=np.float32，如果提供给稀疏矩阵，则将其转换为csr_matrix。
# MAGIC * **check_input**: bool, default=True  
# MAGIC 允许绕过多个输入检查。除非您知道自己做了什么，否则不要使用此参数。

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 返回
# MAGIC 
# MAGIC * **X_leaves**: array-like of shape (n_samples,)  
# MAGIC 对于 X 中的每个数据点 x，返回叶子 x 的索引。叶子的编号在[0, self.tree_.node_count)内，可能在编号上有间隙。

# COMMAND ----------

# MAGIC %md
# MAGIC #### predict
# MAGIC 
# MAGIC **predict(X, check_input=True)**
# MAGIC 
# MAGIC 预测 X 的类或回归值。
# MAGIC 
# MAGIC 对于分类模型，将返回 X 中每个样本的预测类。对于回归模型，将返回基于 X 的预测值。

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 参数
# MAGIC 
# MAGIC * **X**: {array-like, sparse matrix} of shape (n_samples, n_features)  
# MAGIC 训练输入示例。在内部，将会转换为dtype=np.float32，如果提供给稀疏矩阵，则将其转换为csr_matrix。
# MAGIC * **check_input**: bool, default=True  
# MAGIC 允许绕过多个输入检查。除非您知道自己做了什么，否则不要使用此参数。

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 返回
# MAGIC 
# MAGIC * **y**: array-like of shape (n_samples,) or (n_samples, n_outputs)  
# MAGIC 预测类或预测值。
