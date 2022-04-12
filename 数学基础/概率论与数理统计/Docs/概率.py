# Databricks notebook source
# MAGIC %md
# MAGIC # 概率
# MAGIC 
# MAGIC **先验：** 根据统计历史上的经验、常识，当下事件发生的概率。
# MAGIC 
# MAGIC **似然：** 当下事件由果及因发生的概率。
# MAGIC 
# MAGIC **后验：** 当下事件由因及果发生的概率。

# COMMAND ----------

# MAGIC %md
# MAGIC ## 随机变量
# MAGIC 
# MAGIC 一个事件的所有可能的结果组成这个事件的**样本空间**，其中的每一种结果叫做**样本点**。如果对于每一个样本点，都有一个唯一的实数与之对应，则就产生了一个样本点到唯一实数之间的函数，我们称该函数为**随机变量**。
# MAGIC 
# MAGIC 随机变量中的每一个取值及取值的概率被称为**概率分布**。

# COMMAND ----------

# MAGIC %md
# MAGIC ### 离散型随机变量
# MAGIC 
# MAGIC 当一个随机变量的全部可能取值，只有有限个或者可列无穷多个，则称为**离散型随机变量**。

# COMMAND ----------

# MAGIC %md
# MAGIC #### 概率质量函数 (Probability Mass Function，PMF)
# MAGIC 
# MAGIC 离散型随机变量的概率分布可以使用**概率质量函数（分布律）**来描述。
# MAGIC 
# MAGIC **概率质量函数**是离散型随机变量在各特定取值上的概率。
# MAGIC 
# MAGIC 我们通常用大写字母![](https://www.zhihu.com/equation?tex=P)来表示离散型随机变量![](https://www.zhihu.com/equation?tex=X)的概率质量函数，如![](https://www.zhihu.com/equation?tex=P%28x%29)表示离散型随机变量![](https://www.zhihu.com/equation?tex=X)的概率质量函数。![](https://www.zhihu.com/equation?tex=X%3Dx)的概率用![](https://www.zhihu.com/equation?tex=P%28X%3Dx%29)来表示。
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=X)服从![](https://www.zhihu.com/equation?tex=P%28x%29)的概率分布，记作![](https://www.zhihu.com/equation?tex=X%5Csim+P%28x%29)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 概率质量函数的特性
# MAGIC 
# MAGIC * **非负性：**![](https://www.zhihu.com/equation?tex=P%28x_i%29%5Cgeq+0)，![](https://www.zhihu.com/equation?tex=i%3D1%2C2%2C%5Ccdots)
# MAGIC 
# MAGIC * **归一性：**![](https://www.zhihu.com/equation?tex=%5Csum_%7Bi%3D1%7D%5E%5Cinfty+P%28x_i%29%3D1)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 联合概率分布
# MAGIC 
# MAGIC 联合分布描述了多个随机变量的概率分布，是对单一随机变量的自然拓展。联合分布的多个随机变量都定义在同一个样本空间中。

# COMMAND ----------

# MAGIC %md
# MAGIC #### 伯努利试验
# MAGIC 
# MAGIC **伯努利试验（Bernoulli experiment）**是在同样的条件下重复地、相互独立地进行的一种随机试验，其特点是该随机试验只有两种可能结果：**发生**或者**不发生**。我们假设该项试验独立重复地进行了![](https://www.zhihu.com/equation?tex=n)次，那么就称这一系列重复独立的随机试验为**n重伯努利试验**，或称为**伯努利概型**。
# MAGIC 
# MAGIC * 重复试验的相互独立性
# MAGIC 
# MAGIC * 多次试验

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 典例
# MAGIC 
# MAGIC * 连续的n次射击
# MAGIC 
# MAGIC * 连续的掷n次硬币

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 相关定理
# MAGIC 
# MAGIC 设在一次试验中，事件![](https://www.zhihu.com/equation?tex=A)发生的概率为 ![](https://www.zhihu.com/equation?tex=p%280<p<1%29)，则在![](https://www.zhihu.com/equation?tex=n)重伯努利试验中，事件![](https://www.zhihu.com/equation?tex=A)恰好发生 ![](https://www.zhihu.com/equation?tex=k) 次的概率为：![](https://www.zhihu.com/equation?tex=P_n%28k%29%3DC_n%5Ekp%5Ek%281-p%29%5E%7Bn-k%7D%5Cquad%28k%3D0,1,2,%5Ccdots,n%29)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 推论
# MAGIC 
# MAGIC 设在一次试验中，事件![](https://www.zhihu.com/equation?tex=A)首次发生的概率为 ![](https://www.zhihu.com/equation?tex=p%280<p<1%29)，则在伯努利试验序列中，事件![](https://www.zhihu.com/equation?tex=A)在第 ![](https://www.zhihu.com/equation?tex=k) 次试验中才首次发生的概率为：![](https://www.zhihu.com/equation?tex=p%281-p%29%5E%7Bk-1%7D%5Cquad%28k%3D1,2,%5Ccdots%29)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 常用离散分布

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 二项分布
# MAGIC 
# MAGIC 记 ![](https://www.zhihu.com/equation?tex=X) 为n重伯努利试验中成功的事件（记为 ![](https://www.zhihu.com/equation?tex=A) ）的次数，则 ![](https://www.zhihu.com/equation?tex=X%3D0%2C1%2C2%2C%5Ccdots%2Cn.) ![](https://www.zhihu.com/equation?tex=X) 服从二项分布。
# MAGIC 
# MAGIC 记 ![](https://www.zhihu.com/equation?tex=p) 为事件 ![](https://www.zhihu.com/equation?tex=A) 发生的概率， ![](https://www.zhihu.com/equation?tex=X) 的分布列为：
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=P%28X%3Dk%29%3D%5Cleft%28%5Cbegin%7Barray%7D%7Bl%7D%7Bn%7D+%5C%5C+%7Bk%7D%5Cend%7Barray%7D%5Cright%29+p%5E%7Bk%7D%281-p%29%5E%7Bn-k%7D%2C+%5Cquad+k%3D0%2C1%2C+%5Ccdots%2C+n.) 
# MAGIC 
# MAGIC 记：
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=X%5Csim+b%28n%2Cp%29)。
# MAGIC 
# MAGIC 该记号表示随机变量 ![](https://www.zhihu.com/equation?tex=X) 服从参数为 ![](https://www.zhihu.com/equation?tex=n%2Cp) 的二项分布。
# MAGIC 
# MAGIC > 数学期望：![](https://www.zhihu.com/equation?tex=np)  
# MAGIC 方差：![](https://www.zhihu.com/equation?tex=np%281-p%29)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 两点分布
# MAGIC 
# MAGIC 是一种当 ![](https://www.zhihu.com/equation?tex=n%3D1) 时的特殊的二项分布，又名**0-1分布**，**伯努利分布**，用来描述一次伯努利试验中成功的次数 ![](https://www.zhihu.com/equation?tex=X) 。 ![](https://www.zhihu.com/equation?tex=X%3D0%2C1) ![](https://www.zhihu.com/equation?tex=X) 服从两点分布，分布列为：
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=P%28X%3Dx%29%3Dp%5E%7Bx%7D%281-p%29%5E%7B1-x%7D%2C%5Cquad+x%3D0%2C1)
# MAGIC 
# MAGIC 或表示为：
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=%5Cbegin%7Barray%7D%7Bc%7Cc%7D+%5Ctext%7BX%7D+%26+0%26+1+%5C%5C+%5Chline+P+%261-p%26p+%5Cend%7Barray%7D)
# MAGIC 
# MAGIC 其中 ![](https://www.zhihu.com/equation?tex=p%3DP%28X%3D1%29) 为事件成功的概率。
# MAGIC 
# MAGIC > 数学期望：![](https://www.zhihu.com/equation?tex=p)  
# MAGIC 方差：![](https://www.zhihu.com/equation?tex=p%281-p%29)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 负二项分布
# MAGIC 
# MAGIC 在伯努利试验序列中，记每次试验中事件 ![](https://www.zhihu.com/equation?tex=A) 发生的概率为 ![](https://www.zhihu.com/equation?tex=p) ，如果 ![](https://www.zhihu.com/equation?tex=X) 为事件 ![](https://www.zhihu.com/equation?tex=A) 第 ![](https://www.zhihu.com/equation?tex=r) 次出现时的试验次数，则 ![](https://www.zhihu.com/equation?tex=X) 的可能取值为 ![](https://www.zhihu.com/equation?tex=r%2Cr%2B1%2C%5Ccdots%2Cr%2Bm%2C%5Ccdots) ，称![](https://www.zhihu.com/equation?tex=X)服从**负二项分布**或**巴斯卡分布**，其分布列为：
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=P%28X%3Dk%29%3D%5Cleft%28%5Cbegin%7Barray%7D%7Bl%7D%7Bk-1%7D+%5C%5C+%7Br-1%7D%5Cend%7Barray%7D%5Cright%29+p%5Er%5C%2C%281-p%29%5E%7Bk-r%7D%2C+%5Cquad+k%3Dr%2C+r%2B1%2C+%5Ccdots)
# MAGIC 
# MAGIC 记作： ![](https://www.zhihu.com/equation?tex=X%5Csim+Nb%28r%2Cp%29) ，当 ![](https://www.zhihu.com/equation?tex=r%3D1) 时即为**几何分布**，即**几何分布是特殊的负二项分布**。
# MAGIC 
# MAGIC > 数学期望：![](https://www.zhihu.com/equation?tex=%5Cfrac%7Br%7D%7Bp%7D)  
# MAGIC 方差：![](https://www.zhihu.com/equation?tex=%5Cfrac%7Br%281-p%29%7D%7Bp%5E2%7D)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 几何分布
# MAGIC 
# MAGIC 在伯努利试验序列中，记每次试验中事件 ![](https://www.zhihu.com/equation?tex=A) 发生的概率为 ![](https://www.zhihu.com/equation?tex=p) ,如果 ![](https://www.zhihu.com/equation?tex=X) 为事件 ![](https://www.zhihu.com/equation?tex=A) 首次出现时的试验次数，则 ![](https://www.zhihu.com/equation?tex=X%3D1%2C2%2C%5Ccdots)。 ![](https://www.zhihu.com/equation?tex=X) 服从几何分布，分布列为：
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=P%28X%3Dk%29%3D%281-p%29%5E%7Bk-1%7D+p%2C+%5Cquad+k%3D1%2C2%2C+%5Ccdots)
# MAGIC 
# MAGIC 记作 ![](https://www.zhihu.com/equation?tex=X%5Csim+Ge%28p%29)。
# MAGIC 
# MAGIC > 数学期望：![](https://www.zhihu.com/equation?tex=%5Cfrac%7B1%7D%7Bp%7D)  
# MAGIC 方差：![](https://www.zhihu.com/equation?tex=%5Cfrac%7B1-p%7D%7Bp%5E2%7D)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 超几何分布
# MAGIC 
# MAGIC 设有 ![](https://www.zhihu.com/equation?tex=N) 件产品，其中有 ![](https://www.zhihu.com/equation?tex=M) 件不合格品。若从中不放回地随机抽取 ![](https://www.zhihu.com/equation?tex=n) 件，则其中含有的不合格品的件数 ![](https://www.zhihu.com/equation?tex=X) 服从超几何分布，分布列为：
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=P%28X%3Dk%29%3D%5Cfrac%7B%5Cbinom%7BM%7D%7Bk%7D%5Cbinom%7BN-M%7D%7Bn-k%7D+%7D%7B%5Cbinom%7BN%7D%7Bn%7D%7D%2C%5Cquad+k%3D0%2C1%2C+%5Ccdots%2C+r)
# MAGIC 
# MAGIC 记为 ![](https://www.zhihu.com/equation?tex=X%5Csim+h%28n%2CN%2CM%29) .其中 ![](https://www.zhihu.com/equation?tex=r%3D%5Cmin+%5C%7BM%2C+n%5C%7D) ，且 ![](https://www.zhihu.com/equation?tex=M+%5Cleqslant+N%2C+n+%5Cleqslant+N)。 ![](https://www.zhihu.com/equation?tex=n%2C+N%2C+M) 均为正整数。
# MAGIC 
# MAGIC > 数学期望：![](https://www.zhihu.com/equation?tex=n%5Cfrac%7BM%7D%7BN%7D)  
# MAGIC 方差：![](https://www.zhihu.com/equation?tex=%5Cfrac%7BnM%28N-M%29%28N-n%29%7D%7BN%5E2%28N-1%29%7D)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 泊松分布
# MAGIC 
# MAGIC 泊松分布的参数 ![](https://www.zhihu.com/equation?tex=%5Clambda) 是单位时间(或单位面积)内随机事件的平均发生次数。 泊松分布适合于描述单位时间内随机事件发生的次数。
# MAGIC 
# MAGIC 泊松分布的概率函数为：
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=P%28X%3Dk%29%3D%5Cfrac%7B%5Clambda%5E%7Bk%7D%7D%7Bk+%21%7D+%5Cmathrm%7Be%7D%5E%7B-%5Clambda%7D%2C%5Cquad+k%3D0%2C1%2C2%2C+%5Ccdots)
# MAGIC 
# MAGIC 记 ![](https://www.zhihu.com/equation?tex=X%5Csim+P%28%5Clambda%29) 。
# MAGIC 
# MAGIC > 数学期望：![](https://www.zhihu.com/equation?tex=%5Clambda)  
# MAGIC 方差：![](https://www.zhihu.com/equation?tex=%5Clambda)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 连续型随机变量
# MAGIC 
# MAGIC **连续型随机变量**是指如果随机变量![](https://www.zhihu.com/equation?tex=X)的所有可能取值不可以逐个列举出来，而是取数轴上某一区间内的任一点的随机变量。

# COMMAND ----------

# MAGIC %md
# MAGIC #### 概率密度函数
# MAGIC 
# MAGIC 若存在非负可积函数 ![](https://www.zhihu.com/equation?tex=f%28x%29) ，使得随机变量 ![](https://www.zhihu.com/equation?tex=X) 满足：![](https://www.zhihu.com/equation?tex=P%28x%29%3D%5Cint_%7B-%5Cinfty%7D%5E%7Bx%7Df%28t%29%7B%5Crm+d%7Dt)，那么 ![](https://www.zhihu.com/equation?tex=X) 是**连续型随机变量**，![](https://www.zhihu.com/equation?tex=f%28x%29) 是 ![](https://www.zhihu.com/equation?tex=X) 的**概率密度函数**。

# COMMAND ----------

# MAGIC %md
# MAGIC #### 概率密度函数的特性
# MAGIC 
# MAGIC 密度函数![](https://www.zhihu.com/equation?tex=f%28x%29)具有下列性质：
# MAGIC 
# MAGIC 1、![](https://www.zhihu.com/equation?tex=f%28x%29%5Cgeq0)
# MAGIC 
# MAGIC 2、![](https://www.zhihu.com/equation?tex=%5Cint_%7B-%5Cinfty%7D%5E%7B%2B%5Cinfty%7Df%28x%29%7B%5Crm+d%7Dx%3D1)
# MAGIC 
# MAGIC 3、![](https://www.zhihu.com/equation?tex=P%28a%5Clt+x%5Cleq+b%29%3D%5Cint_%7Ba%7D%5E%7Bb%7Df%28x%29%7B%5Crm+d%7Dx)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 常用连续分布

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 正态分布
# MAGIC 
# MAGIC 若随机变量 ![](https://www.zhihu.com/equation?tex=X) 的密度函数为：
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=p%28x%29%3D%5Cfrac%7B1%7D%7B%5Csqrt%7B2+%5Cpi%7D+%5Csigma%7D+%5Cmathrm%7Be%7D%5E%7B-%5Cfrac%7B%28x-%5Cmu%29%5E%7B2%7D%7D%7B2+%5Csigma%5E%7B2%7D%7D%7D%2C%5Cquad+-%5Cinfty%3Cx%3C%5Cinfty)
# MAGIC 
# MAGIC 则称 ![](https://www.zhihu.com/equation?tex=X) 服从**正态分布**，称 ![](https://www.zhihu.com/equation?tex=X) 为**正态变量**。
# MAGIC 
# MAGIC 若随机变量 ![](https://www.zhihu.com/equation?tex=X) 服从一个**数学期望（或均值，亦简称期望）**为![](https://www.zhihu.com/equation?tex=%5Cmu)、**方差**为![](https://www.zhihu.com/equation?tex=%5Csigma%5E2)的正态分布，记为![](https://www.zhihu.com/equation?tex=X%5Csim+N%28%5Cmu%2C%5Csigma%5E2%29)。其中![](https://www.zhihu.com/equation?tex=%5Cmu)为位置参数，用于控制曲线在 ![](https://www.zhihu.com/equation?tex=x) 轴上的位置；![](https://www.zhihu.com/equation?tex=%5Csigma)为尺度参数，用于控制曲线的形状。
# MAGIC 
# MAGIC 分布函数：
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=F%28x%29%3D%5Cint_%7B-%5Cinfty%7D%5E%7Bx%7Dp%28t%29%7B%5Crm+d%7Dt%3D%5Cint_%7B-%5Cinfty%7D%5E%7Bx%7D%5Cfrac%7B1%7D%7B%5Csqrt%7B2+%5Cpi%7D+%5Csigma%7D+%5Cmathrm%7Be%7D%5E%7B-%5Cfrac%7B%28t-%5Cmu%29%5E%7B2%7D%7D%7B2+%5Csigma%5E%7B2%7D%7D%7D%7B%5Crm+d%7Dt)
# MAGIC 
# MAGIC > 数学期望：![](https://www.zhihu.com/equation?tex=%5Cmu)  
# MAGIC 方差：![](https://www.zhihu.com/equation?tex=%5Csigma%5E2)
# MAGIC 
# MAGIC **一些常用性质：**
# MAGIC 
# MAGIC 若 ![](https://www.zhihu.com/equation?tex=X%5Csim+N%28%5Cmu%2C%5Csigma%5E2%29) ：
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=P%28X%5Cleq+c%29%3D%5CPhi%5Cleft%28%5Cfrac%7Ba-%5Cmu%7D%7B%5Csigma%7D%5Cright%29)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=P%28a%3Cx%5Cleq+b%29%3D%5CPhi%5Cleft%28%5Cfrac%7Bb-%5Cmu%7D%7B%5Csigma%7D%5Cright%29-%5CPhi%5Cleft%28%5Cfrac%7Ba-%5Cmu%7D%7B%5Csigma%7D%5Cright%29)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 标准正态分布
# MAGIC 
# MAGIC 称 ![](https://www.zhihu.com/equation?tex=%5Cmu%3D0%2C%5Csigma%5E2%3D1) 时的正态分布为**标准正态分布**，记为 ![](https://www.zhihu.com/equation?tex=X%5Csim+N%280%2C1%29)，其
# MAGIC 密度函数为：
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=%5Cvarphi%28x%29%3D%5Cfrac%7B1%7D%7B%5Csqrt%7B2%5Cpi%7D%7D%5C%2C%7B%5Crm+e%7D%5E%7B-%5Cfrac%7Bx%5E2%7D%7B2%7D%7D)
# MAGIC 
# MAGIC 分布函数为：
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=%5CPhi%28x%29%3D%5Cint_%7B-%5Cinfty%7D%5E%7Bx%7D%5Cvarphi%28t%29%7B%5Crm+d%7Dt%3D%5Cint_%7B-%5Cinfty%7D%5E%7Bx%7D%5Cfrac%7B1%7D%7B%5Csqrt%7B2%5Cpi%7D%7D%5C%2C%7B%5Crm+e%7D%5E%7B-%5Cfrac%7Bt%5E2%7D%7B2%7D%7D%7B%5Crm+d%7Dt)
# MAGIC 
# MAGIC 任何一个正态变量均可以通过**标准化**转化为**标准正态变量**，即若 ![](https://www.zhihu.com/equation?tex=X%5Csim+N%28%5Cmu%2C%5Csigma%5E2%29) ，则：
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=X%5E%2A%3D%5Cfrac%7BX-%5Cmu%7D%7B%5Csigma%7D%5Csim+N%280%2C1%29)
# MAGIC 
# MAGIC 其中 ![](https://www.zhihu.com/equation?tex=X%5E%2A) 为**标准正态变量**。
# MAGIC 
# MAGIC 
# MAGIC **一些常用性质：**
# MAGIC 
# MAGIC 若 ![](https://www.zhihu.com/equation?tex=X%5Csim+N%280%2C1%29) ：
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=%5CPhi%28-a%29%3D1-%5CPhi%28a%29)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=P%28X%3Ea%29%3D1-%5CPhi%28a%29)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=P%28a%3Cx%3Cb%29%3D%5CPhi%28b%29-%5CPhi%28a%29)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=P%28%5C%2C%7CX%7C%3Cc%29%3D2%5CPhi%28c%29-1%2C%5Cquad+%28c%5Cgeq0%29)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 卡方分布
# MAGIC 
# MAGIC 若![](https://www.zhihu.com/equation?tex=n)个相互独立的随机变量![](https://www.zhihu.com/equation?tex=%5Cxi_1,%5Cxi_2,%5Cdots,%5Cxi_n)，均服从**标准正态分布**（也称独立同分布于标准正态分布），则这![](https://www.zhihu.com/equation?tex=n)个服从标准正态分布的随机变量的平方和![](https://www.zhihu.com/equation?tex=Q%3D%5Csum_%7Bi%3D1%7D^%7Bn%7D%5Cxi_i^2)构成一新的随机变量，其分布规律称为**卡方分布**（![](https://www.zhihu.com/equation?tex=%5Cchi^2)，chi-square distribution），其中参数 ![](https://www.zhihu.com/equation?tex=n%3Dv)，称为**自由度**，正如正态分布中期望或方差不同就是另一个正态分布一样，自由度不同就是另一个![](https://www.zhihu.com/equation?tex=%5Cchi^2)分布。记为![](https://www.zhihu.com/equation?tex=X%5Csim+%5Cchi^2%28v%29)或者![](https://www.zhihu.com/equation?tex=X%5Csim+%5Cchi^2_v)（其中![](https://www.zhihu.com/equation?tex=v%3Dn-k)，![](https://www.zhihu.com/equation?tex=k)为限制条件数）。
# MAGIC 
# MAGIC 卡方分布是由正态分布构造而成的一个新的分布，当自由度![](https://www.zhihu.com/equation?tex=v)很大时，![](https://www.zhihu.com/equation?tex=%5Cchi^2)分布近似为正态分布。
# MAGIC 
# MAGIC 对于任意正整数![](https://www.zhihu.com/equation?tex=x)， 自由度为![](https://www.zhihu.com/equation?tex=v)的卡方分布是一个随机变量![](https://www.zhihu.com/equation?tex=X)的机率分布。
# MAGIC 
# MAGIC > 数学期望： 自由度![](https://www.zhihu.com/equation?tex=v)，![](https://www.zhihu.com/equation?tex=E%28%5Cchi^2%28v%29%29%3Dv)  
# MAGIC 方差： 自由度![](https://www.zhihu.com/equation?tex=v)的2倍，![](https://www.zhihu.com/equation?tex=D%28%5Cchi^2%29%3D2v) 

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 均匀分布
# MAGIC 
# MAGIC 若随机变量 ![](https://www.zhihu.com/equation?tex=X) 的密度函数为：
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=p%28x%29%3D%5Cbegin%7Bcases%7D+%5Cfrac%7B1%7D%7Bb-a%7D%2C%5Cquad+a%3Cx%3Cb%5C%5C+0%2C%5Cquad%5Cquad%5Cquad+%E5%85%B6%E4%BB%96%5Cend%7Bcases%7D)
# MAGIC 
# MAGIC 称 ![](https://www.zhihu.com/equation?tex=X) 服从区间 ![](https://www.zhihu.com/equation?tex=%28a%2Cb%29) 上的**均匀分布**，记作 ![](https://www.zhihu.com/equation?tex=X%5Csim+U%28a%2Cb%29) ，其分布函数：
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=F%28x%29%3D%5Cbegin%7Bcases%7D+0%2C%5Cquad+%5Cquad+%5C%2Cx%3Ca%2C%5C%5C+%5Cfrac%7Bx-a%7D%7Bb-a%7D%2C%5Cquad+a%5Cleq+x%3Cb%2C%5C%5C+1%2C%5Cquad+%5Cquad+%5C%2Cx%5Cgeq+b.++%5Cend%7Bcases%7D)
# MAGIC 
# MAGIC 均匀分布又称作**平顶分布**（因其概率密度为常值函数）。
# MAGIC 
# MAGIC > 数学期望： ![](https://www.zhihu.com/equation?tex=%5Cfrac%7Ba%2Bb%7D%7B2%7D)  
# MAGIC 方差： ![](https://www.zhihu.com/equation?tex=%5Cfrac%7B%28b-a%29%5E2%7D%7B12%7D) 

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 指数分布
# MAGIC 
# MAGIC 若随机变量 ![](https://www.zhihu.com/equation?tex=X) 的密度函数为：
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=p%28x%29%3D%5Cbegin%7Bcases%7D+%5Clambda%7B%5Crm+e%7D%5E%7B-%5Clambda+x%7D%2C%5Cquad+x%5Cgeq0%5C%5C+%5Cquad+0%2C%5Cquad+%5C%2C%5C%2C%5C%2Cx%3C0+%5Cend%7Bcases%7D)
# MAGIC 
# MAGIC 称 ![](https://www.zhihu.com/equation?tex=X) 服从参数为 ![](https://www.zhihu.com/equation?tex=%5Clambda) 的**指数分布**，记作 ![](https://www.zhihu.com/equation?tex=X%5Csim++Exp%28%5Clambda%29) ，其分布函数：
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=F%28x%29%3D%5Cbegin%7Bcases%7D+1-%7B%5Crm+e%7D%5E%7B%5Clambda+x%7D%2C%5Cquad+x%5Cgeq0%5C%5C+%5Cquad+%5C+0%2C%5Cquad%5Cquad+%5C%2C%5C%2C+x%3C0+%5Cend%7Bcases%7D)
# MAGIC 
# MAGIC 指数分布是一种偏态分布，指数分布随机变量只可能取非负实数。指数分布常被用作各种“**寿命**”分布，譬如**电子元器件的寿命**、**动物的寿命**、**电话的通话时间**、**随机服务系统中的服务时间**等都可假定服从指数分布。指数分布在**可靠性**与**排队论**中有着广泛的应用。
# MAGIC 
# MAGIC > 数学期望： ![](https://www.zhihu.com/equation?tex=%5Cfrac%7B1%7D%5Clambda)  
# MAGIC 方差： ![](https://www.zhihu.com/equation?tex=%5Cfrac%7B1%7D%7B%5Clambda%5E2%7D) 

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 伽玛分布
# MAGIC 
# MAGIC **伽玛函数：**
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=%5CGamma+%28+%5Calpha+%29+%3D+%5Cint+_+%7B+0+%7D+%5E+%7B+%5Cinfty+%7D+x+%5E+%7B+%5Calpha+-+1+%7D+e+%5E+%7B+-+x+%7D+d+x)
# MAGIC 
# MAGIC 其中参数 ![](https://www.zhihu.com/equation?tex=%5Calpha%3E0) 。
# MAGIC 
# MAGIC 伽玛函数具有下列**性质：**
# MAGIC 
# MAGIC > ![](https://www.zhihu.com/equation?tex=%5CGamma%281%29%3D1%2C%5Cquad+%5CGamma%28%5Cfrac%7B1%7D%7B2%7D%29%3D%5Csqrt+%7B%5Cpi%7D)  
# MAGIC ![](https://www.zhihu.com/equation?tex=%5CGamma%28%5Calpha%2B1%29%3D%5Calpha%5CGamma%28%5Calpha%29)
# MAGIC 
# MAGIC 当 ![](https://www.zhihu.com/equation?tex=%5Calpha) 为自然数 ![](https://www.zhihu.com/equation?tex=n) 时：
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=%5CGamma%28n%2B1%29%3Dn%5CGamma%28n%29%3Dn%5C%2C%21)
# MAGIC 
# MAGIC **伽玛分布：**
# MAGIC 
# MAGIC 若随机变量 ![](https://www.zhihu.com/equation?tex=X) 的密度函数为：
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=p+%28+x+%29+%3D+%5Cleft%5C%7B+%5Cbegin%7Barray%7D+%7B+l+l+%7D+%7B+%5Cfrac+%7B+%5Clambda+%5E+%7B+a+%7D+%7D+%7B+%5CGamma+%28+%5Calpha+%29+%7D+x+%5E+%7B+a+-+1+%7D+%5Cmathrm+%7B+e+%7D+%5E+%7B+-+%5Clambda+x+%7D+%2C+%7D+%26+%7B+x+%5Cgeqslant+0%7D+%5C%5C+%7B+%5Cquad+%5Cquad+%5Cquad0%2C+++%7D+%26+%7B+x+%3C+0+%7D+%5Cend%7Barray%7D+%5Cright.)
# MAGIC 
# MAGIC 称 ![](https://www.zhihu.com/equation?tex=X) 服从**伽玛分布**，记作 ![](https://www.zhihu.com/equation?tex=X%5Csim+Ga%28%5Calpha%2C%5Clambda%29) 。其中 ![](https://www.zhihu.com/equation?tex=%5Calpha%3E0) 为**形状参数**， ![](https://www.zhihu.com/equation?tex=%5Clambda%3E0) 为**尺度参数**。
# MAGIC 
# MAGIC > 数学期望： ![](https://www.zhihu.com/equation?tex=%5Cfrac%7B%5Calpha%7D%7B%5Clambda%7D)  
# MAGIC 方差： ![](https://www.zhihu.com/equation?tex=%5Cfrac%7B%5Calpha%7D%7B%5Clambda%5E2%7D) 

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 贝塔分布
# MAGIC 
# MAGIC **贝塔函数：**
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=%5Cmathrm+%7B+B+%7D+%28+a+%2C+b+%29+%3D+%5Cint+_+%7B+0+%7D+%5E+%7B+1+%7D+x+%5E+%7B+a+-+1+%7D+%28+1+-+x+%29+%5E+%7B+b+-+1+%7D+d+x)
# MAGIC 
# MAGIC 其中参数 ![](https://www.zhihu.com/equation?tex=a%3E0%2Cb%3E0) 。
# MAGIC 
# MAGIC 贝塔函数具有以下**性质：**
# MAGIC 
# MAGIC > 1、![](https://www.zhihu.com/equation?tex=B%28a%2Cb%29%3DB%28b%2Ca%29)  
# MAGIC 2、贝塔函数与伽玛函数有如下关系：![](https://www.zhihu.com/equation?tex=%5Cmathrm+%7B+B+%7D+%28+a+%2C+b+%29+%3D+%5Cfrac+%7B+%5CGamma+%28+a+%29+%5C%2C%5CGamma+%28+b+%29+%7D+%7B+%5CGamma+%28+a+%2B+b+%29+%7D)
# MAGIC 
# MAGIC **贝塔分布：**
# MAGIC 
# MAGIC 若随机变量 ![](https://www.zhihu.com/equation?tex=X) 的密度函数为：
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=p+%28+x+%29+%3D+%5Cleft%5C%7B+%5Cbegin%7Barray%7D+%7B+l+l+%7D+%7B%5Cfrac+%7B+%5CGamma+%28+a%2Bb+%29++%7D+%7B+%5CGamma+%28+a++%29+%5CGamma+%28+b+%29%7D+x+%5E+%7B+a+-+1+%7D+%281-x%29%5E%7Bb-1%7D+%7D%2C+%26+%7B0%3Cx%3C1%7D+%5C%5C+%7B+%5Cquad+%5Cquad+%5Cquad0%2C+++%7D+%26+%7B+%5Cquad+%E5%85%B6%E4%BB%96+%7D+%5Cend%7Barray%7D+%5Cright.)
# MAGIC 
# MAGIC 则称 ![](https://www.zhihu.com/equation?tex=X) 服从**贝塔分布**，记作 ![](https://www.zhihu.com/equation?tex=X%5Csim+Be%28a%2Cb%29) ，其中 ![](https://www.zhihu.com/equation?tex=a%3E0%2Cb%3E0) 都是**形状参数**。
# MAGIC 
# MAGIC > 数学期望： ![](https://www.zhihu.com/equation?tex=%5Cfrac%7Ba%7D%7Ba%2Bb%7D)  
# MAGIC 方差： ![](https://www.zhihu.com/equation?tex=%5Cfrac%7Bab%7D%7B%28a%2Bb%29%5E2%28a%2Bb%2B1%29%7D) 

# COMMAND ----------

# MAGIC %md
# MAGIC # 条件概率
# MAGIC 
# MAGIC 假设我们知道给定事件![](https://www.zhihu.com/equation?tex=B)已经发生，在此基础上希望知道另一个事件![](https://www.zhihu.com/equation?tex=A)发生的可能性，此时我们就需要构造出条件概率，它需要先顾及事件![](https://www.zhihu.com/equation?tex=B)已经发生的信息，然后再求出事件![](https://www.zhihu.com/equation?tex=A)发生的概率。
# MAGIC 
# MAGIC **条件概率**描述的就是在给定事件![](https://www.zhihu.com/equation?tex=B)发生的情况下，事件![](https://www.zhihu.com/equation?tex=A)发生的概率，我们专门把他记作： ![](https://www.zhihu.com/equation?tex=P%28A%7CB%29)。并且：
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=P%28A%7CB%29+%3D+%5Cfrac%7BP%28AB%29%7D%7BP%28B%29%7D)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 乘法公式
# MAGIC 
# MAGIC 根据条件概率的公式，我们可以得到如下等式：
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=P%28AB%29%3DP%28A%7CB%29P%28B%29%3DP%28B%7CA%29P%28A%29)

# COMMAND ----------

# MAGIC %md
# MAGIC # 全概率公式
# MAGIC 
# MAGIC 如果事件![](https://www.zhihu.com/equation?tex=B_1,B_2,%5Ccdots)满足：
# MAGIC 
# MAGIC * ![](https://www.zhihu.com/equation?tex=B_1,B_2,%5Ccdots) 两两互斥，也就是![](https://www.zhihu.com/equation?tex=B_i%5Ccap+B_j%3D%5Cphi)，![](https://www.zhihu.com/equation?tex=i%5Cne+j)，![](https://www.zhihu.com/equation?tex=i,j%3D1,2,%5Ccdots)，且![](https://www.zhihu.com/equation?tex=P%28B_i%29%5Cgt0)，![](https://www.zhihu.com/equation?tex=i%3D1%2C2%2C%5Ccdots)
# MAGIC * ![](https://www.zhihu.com/equation?tex=B_1%2BB_2%5Ccdots%3D%5COmega)，则称事件组![](https://www.zhihu.com/equation?tex=B_1,B_2,%5Ccdots)是样本空间![](https://www.zhihu.com/equation?tex=%5COmega)的一个划分。设![](https://www.zhihu.com/equation?tex=B_1,B_2,%5Ccdots)是样本空间![](https://www.zhihu.com/equation?tex=%5COmega)的一个划分，![](https://www.zhihu.com/equation?tex=A)是任一事件。
# MAGIC 
# MAGIC 则：
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=P%28A%29%3DP%28B_1%29P%28A%7CB_1%29%2BP%28B_2%29P%28A%7CB_2%29%2B...%2BP%28B_n%29P%28A%7CB_n%29%3D%5Csum_%7Bi%3D1%7D%5En+P%28B_i%29P%28A%7CB_i%29)

# COMMAND ----------

# MAGIC %md
# MAGIC # 贝叶斯公式
# MAGIC 
# MAGIC 与全概率公式解决的问题相反，贝叶斯公式是建立在条件概率的基础上寻找事件发生的原因（即大事件![](https://www.zhihu.com/equation?tex=A)已经发生的条件下，分割中的小事件![](https://www.zhihu.com/equation?tex=B_i)发生的概率）。
# MAGIC 
# MAGIC 设![](https://www.zhihu.com/equation?tex=B_1,B_2,%5Ccdots)是样本空间![](https://www.zhihu.com/equation?tex=%5COmega)的一个划分，则对任一事件![](https://www.zhihu.com/equation?tex=A)，![](https://www.zhihu.com/equation?tex=P%28A%29%5Cgt0)，有：
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=P%28B_i%7CA%29%3D%5Cfrac%7BP%28A%7CB_i%29P%28B_i%29%7D%7BP%28A%29%7D%3D%5Cfrac%7BP%28A%7CB_i%29P%28B_i%29%7D%7B%5Csum_%7Bj%3D1%7D%5En+P%28B_j%29P%28A%7CB_j%29%7D)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=B_i)常被视为导致试验结果![](https://www.zhihu.com/equation?tex=A)发生的“原因”，![](https://www.zhihu.com/equation?tex=P%28B_i%29)，![](https://www.zhihu.com/equation?tex=i%3D1%2C2%2C%5Ccdots)，表示各种原因发生的可能性大小，故称**先验概率**。
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=P%28B_i%7CA%29)，![](https://www.zhihu.com/equation?tex=i%3D1%2C2%2C%5Ccdots)，则反映当试验产生了结果![](https://www.zhihu.com/equation?tex=A)之后，再对各种原因概率的重新认识，故称**后验概率**。

# COMMAND ----------

# MAGIC %md
# MAGIC ## 朴素贝叶斯

# COMMAND ----------

# MAGIC %md
# MAGIC ### 朴素贝叶斯分类
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=P%28%E7%B1%BB%E5%88%AB%7C%E7%89%B9%E5%BE%81%29%3D%5Cfrac%7BP%28%E7%89%B9%E5%BE%81%7C%E7%B1%BB%E5%88%AB%29P%28%E7%B1%BB%E5%88%AB%29%7D%7BP%28%E7%89%B9%E5%BE%81%29%7D)

# COMMAND ----------

# MAGIC %md
# MAGIC ### sklearn朴素贝叶斯

# COMMAND ----------

from sklearn.naive_bayes import BernoulliNB #伯努利型
from sklearn.naive_bayes import GaussianNB #高斯分布型
from sklearn.naive_bayes import MultinomialNB #多项式型

from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
from sklearn import datasets

# 鸢尾花数据集
iris=load_iris()
print(iris)

# COMMAND ----------

gnb=BernoulliNB()
# 交叉验证
scores=cross_val_score(gnb,iris.data,iris.target,cv=10) #将数据集分为10份，其中9份作为训练模型，1份用来做评估
                                                        #score是交叉验证的对象
                                                        #结果是返回Accuracy准确率的概念，结果是33.3%
print("Accuracy:%.3f" % scores.mean())

# COMMAND ----------

gnb=GaussianNB()
# 交叉验证
scores=cross_val_score(gnb,iris.data,iris.target,cv=10)
print("Accuracy:%.3f" % scores.mean())

# COMMAND ----------

gnb=MultinomialNB()
# 交叉验证
scores=cross_val_score(gnb,iris.data,iris.target,cv=10) 
print("Accuracy:%.3f" % scores.mean())
