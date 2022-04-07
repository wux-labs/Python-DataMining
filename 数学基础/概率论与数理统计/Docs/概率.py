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
# MAGIC **伯努利试验（Bernoulli experiment）**是在同样的条件下重复地、相互独立地进行的一种随机试验，其特点是该随机试验只有两种可能结果：**发生**或者**不发生**。我们假设该项试验独立重复地进行了n次，那么就称这一系列重复独立的随机试验为**n重伯努利试验**，或称为**伯努利概型**。
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
# MAGIC 设在一次试验中，事件![](https://www.zhihu.com/equation?tex=A)发生的概率为 p（0<p<1），则在n重伯努利试验中，事件![](https://www.zhihu.com/equation?tex=A)恰好发生 k 次的概率为：![](https://www.zhihu.com/equation?tex=P_n%28k%29%3DC_n%5Ekp%5Ek%281-p%29%5E%7Bn-k%7D%5Cquad%28k%3D0,1,2,%5Ccdots,n%29)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 推论
# MAGIC 
# MAGIC 设在一次试验中，事件![](https://www.zhihu.com/equation?tex=A)首次发生的概率为 p（0<p<1），则在伯努利试验序列中，事件![](https://www.zhihu.com/equation?tex=A)在第 k 次试验中才首次发生的概率为：![](https://www.zhihu.com/equation?tex=p%281-p%29%5E%7Bk-1%7D%5Cquad%28k%3D1,2,%5Ccdots%29)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 常用离散分布

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 二项分布
# MAGIC 
# MAGIC 记 ![](https://www.zhihu.com/equation?tex=X) 为n重伯努利试验中成功的事件（记为 ![](https://www.zhihu.com/equation?tex=A) ）的次数，则 ![](https://gitee.com/wux_labs/Data-Scientist/raw/master/数学基础/概率论与数理统计/概率/20220302230432365.png) ![](https://www.zhihu.com/equation?tex=X) 服从二项分布。
# MAGIC 
# MAGIC 记 ![](https://gitee.com/wux_labs/Data-Scientist/raw/master/数学基础/概率论与数理统计/概率/20220302230610272.png) 为事件 ![](https://www.zhihu.com/equation?tex=A) 发生的概率， ![](https://www.zhihu.com/equation?tex=X) 的分布列为：
# MAGIC 
# MAGIC ![](https://gitee.com/wux_labs/Data-Scientist/raw/master/数学基础/概率论与数理统计/概率/20220302230920726.png) 
# MAGIC 
# MAGIC 记：
# MAGIC 
# MAGIC ![](https://gitee.com/wux_labs/Data-Scientist/raw/master/数学基础/概率论与数理统计/概率/20220302231004856.png)。
# MAGIC 
# MAGIC 该记号表示随机变量 ![](https://www.zhihu.com/equation?tex=X) 服从参数为 ![](https://gitee.com/wux_labs/Data-Scientist/raw/master/数学基础/概率论与数理统计/概率/20220302231146911.png) 的二项分布。
# MAGIC 
# MAGIC > 数学期望：![](https://gitee.com/wux_labs/Data-Scientist/raw/master/数学基础/概率论与数理统计/概率/20220302231501627.png)  
# MAGIC 方差：![](https://gitee.com/wux_labs/Data-Scientist/raw/master/数学基础/概率论与数理统计/概率/20220302231550323.png)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 两点分布
# MAGIC 
# MAGIC 是一种当 ![](https://gitee.com/wux_labs/Data-Scientist/raw/master/数学基础/概率论与数理统计/概率/20220302231910512.png) 时的特殊的二项分布，又名**0-1分布**，**伯努利分布**，用来描述一次伯努利试验中成功的次数 ![](https://www.zhihu.com/equation?tex=X) 。 ![](https://gitee.com/wux_labs/Data-Scientist/raw/master/数学基础/概率论与数理统计/概率/20220302232058891.png) ![](https://www.zhihu.com/equation?tex=X) 服从两点分布，分布列为：
# MAGIC 
# MAGIC ![](https://gitee.com/wux_labs/Data-Scientist/raw/master/数学基础/概率论与数理统计/概率/20220302232256319.png)
# MAGIC 
# MAGIC 或表示为：
# MAGIC 
# MAGIC ![](https://gitee.com/wux_labs/Data-Scientist/raw/master/数学基础/概率论与数理统计/概率/20220302232308889.png)
# MAGIC 
# MAGIC 其中 ![](https://gitee.com/wux_labs/Data-Scientist/raw/master/数学基础/概率论与数理统计/概率/20220302232417102.png) 为事件成功的概率。
# MAGIC 
# MAGIC > 数学期望：![](https://gitee.com/wux_labs/Data-Scientist/raw/master/数学基础/概率论与数理统计/概率/20220302230610272.png)  
# MAGIC 方差：![](https://gitee.com/wux_labs/Data-Scientist/raw/master/数学基础/概率论与数理统计/概率/20220302232611540.png)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 负二项分布
# MAGIC 
# MAGIC 在伯努利试验序列中，记每次试验中事件 ![](https://www.zhihu.com/equation?tex=A) 发生的概率为 ![](https://gitee.com/wux_labs/Data-Scientist/raw/master/数学基础/概率论与数理统计/概率/20220302230610272.png) ，如果 ![](https://www.zhihu.com/equation?tex=X) 为事件 ![](https://www.zhihu.com/equation?tex=A) 第 ![](https://gitee.com/wux_labs/Data-Scientist/raw/master/数学基础/概率论与数理统计/概率/20220302233330942.png) 次出现时的试验次数，则 ![](https://www.zhihu.com/equation?tex=X) 的可能取值为 ![](https://gitee.com/wux_labs/Data-Scientist/raw/master/数学基础/概率论与数理统计/概率/20220302233417412.png) ，称![](https://www.zhihu.com/equation?tex=X)服从**负二项分布**或**巴斯卡分布**，其分布列为：
# MAGIC 
# MAGIC ![](https://gitee.com/wux_labs/Data-Scientist/raw/master/数学基础/概率论与数理统计/概率/20220302233436492.png)
# MAGIC 
# MAGIC 记作： ![](https://gitee.com/wux_labs/Data-Scientist/raw/master/数学基础/概率论与数理统计/概率/20220302233530660.png) ，当 ![](https://gitee.com/wux_labs/Data-Scientist/raw/master/数学基础/概率论与数理统计/概率/20220302233549629.png) 时即为**几何分布**，即**几何分布是特殊的负二项分布**。
# MAGIC 
# MAGIC > 数学期望：![](https://gitee.com/wux_labs/Data-Scientist/raw/master/数学基础/概率论与数理统计/概率/20220302234421432.png)  
# MAGIC 方差：![](https://gitee.com/wux_labs/Data-Scientist/raw/master/数学基础/概率论与数理统计/概率/20220302234435112.png)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 几何分布
# MAGIC 
# MAGIC 在伯努利试验序列中，记每次试验中事件 ![](https://www.zhihu.com/equation?tex=A) 发生的概率为 ![](https://gitee.com/wux_labs/Data-Scientist/raw/master/数学基础/概率论与数理统计/概率/20220302230610272.png) ,如果 ![](https://www.zhihu.com/equation?tex=X) 为事件 ![](https://www.zhihu.com/equation?tex=A) 首次出现时的试验次数，则 ![](https://gitee.com/wux_labs/Data-Scientist/raw/master/数学基础/概率论与数理统计/概率/20220302234058603.png)。 ![](https://www.zhihu.com/equation?tex=X) 服从几何分布，分布列为：
# MAGIC 
# MAGIC ![](https://gitee.com/wux_labs/Data-Scientist/raw/master/数学基础/概率论与数理统计/概率/20220302234202500.png)
# MAGIC 
# MAGIC 记作 ![](https://gitee.com/wux_labs/Data-Scientist/raw/master/数学基础/概率论与数理统计/概率/20220302234233010.png)。
# MAGIC 
# MAGIC > 数学期望：![](https://gitee.com/wux_labs/Data-Scientist/raw/master/数学基础/概率论与数理统计/概率/20220302234522922.png)  
# MAGIC 方差：![](https://gitee.com/wux_labs/Data-Scientist/raw/master/数学基础/概率论与数理统计/概率/20220302234537129.png)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 超几何分布
# MAGIC 
# MAGIC 设有 N 件产品，其中有 M 件不合格品。若从中不放回地随机抽取 n 件，则其中含有的不合格品的件数 ![](https://www.zhihu.com/equation?tex=X) 服从超几何分布，分布列为：
# MAGIC 
# MAGIC ![](https://gitee.com/wux_labs/Data-Scientist/raw/master/数学基础/概率论与数理统计/概率/20220303162232234.png)
# MAGIC 
# MAGIC 记为 ![](https://gitee.com/wux_labs/Data-Scientist/raw/master/数学基础/概率论与数理统计/概率/20220303162250410.png) .其中 ![](https://gitee.com/wux_labs/Data-Scientist/raw/master/数学基础/概率论与数理统计/概率/20220303162305363.png) ，且 ![](https://gitee.com/wux_labs/Data-Scientist/raw/master/数学基础/概率论与数理统计/概率/20220303162316783.png)。 ![](https://gitee.com/wux_labs/Data-Scientist/raw/master/数学基础/概率论与数理统计/概率/20220303162327675.png) 均为正整数。
# MAGIC 
# MAGIC > 数学期望：![](https://gitee.com/wux_labs/Data-Scientist/raw/master/数学基础/概率论与数理统计/概率/20220303162516103.png)  
# MAGIC 方差：![](https://gitee.com/wux_labs/Data-Scientist/raw/master/数学基础/概率论与数理统计/概率/20220303162529674.png)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 泊松分布
# MAGIC 
# MAGIC 泊松分布的参数 ![](https://gitee.com/wux_labs/Data-Scientist/raw/master/数学基础/概率论与数理统计/概率/20220303163125627.png) 是单位时间(或单位面积)内随机事件的平均发生次数。 泊松分布适合于描述单位时间内随机事件发生的次数。
# MAGIC 
# MAGIC 泊松分布的概率函数为：
# MAGIC 
# MAGIC ![](https://gitee.com/wux_labs/Data-Scientist/raw/master/数学基础/概率论与数理统计/概率/20220303162915919.png)
# MAGIC 
# MAGIC 记 ![](https://gitee.com/wux_labs/Data-Scientist/raw/master/数学基础/概率论与数理统计/概率/20220303163010119.png) 。
# MAGIC 
# MAGIC > 数学期望：![](https://gitee.com/wux_labs/Data-Scientist/raw/master/数学基础/概率论与数理统计/概率/20220303163125627.png)  
# MAGIC 方差：![](https://gitee.com/wux_labs/Data-Scientist/raw/master/数学基础/概率论与数理统计/概率/20220303163125627.png)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 连续型随机变量
# MAGIC 
# MAGIC **连续型随机变量**是指如果随机变量![](https://www.zhihu.com/equation?tex=X)的所有可能取值不可以逐个列举出来，而是取数轴上某一区间内的任一点的随机变量。

# COMMAND ----------

# MAGIC %md
# MAGIC #### 概率密度函数
# MAGIC 
# MAGIC 若存在非负可积函数 ![](https://gitee.com/wux_labs/Data-Scientist/raw/master/数学基础/概率论与数理统计/概率/20220303172036124.png) ，使得随机变量 ![](https://www.zhihu.com/equation?tex=X) 满足：![](https://gitee.com/wux_labs/Data-Scientist/raw/master/数学基础/概率论与数理统计/概率/20220303172057003.png)，那么 ![](https://www.zhihu.com/equation?tex=X) 是**连续型随机变量**，![](https://gitee.com/wux_labs/Data-Scientist/raw/master/数学基础/概率论与数理统计/概率/20220303172036124.png) 是 ![](https://www.zhihu.com/equation?tex=X) 的**概率密度函数**。

# COMMAND ----------

# MAGIC %md
# MAGIC #### 概率密度函数的特性
# MAGIC 
# MAGIC 密度函数![](https://gitee.com/wux_labs/Data-Scientist/raw/master/数学基础/概率论与数理统计/概率/20220303172036124.png)具有下列性质：
# MAGIC 
# MAGIC 1、![](https://gitee.com/wux_labs/Data-Scientist/raw/master/数学基础/概率论与数理统计/概率/20220303173448053.png)
# MAGIC 
# MAGIC 2、![](https://gitee.com/wux_labs/Data-Scientist/raw/master/数学基础/概率论与数理统计/概率/20220303173500036.png)
# MAGIC 
# MAGIC 3、![](https://gitee.com/wux_labs/Data-Scientist/raw/master/数学基础/概率论与数理统计/概率/20220303173509229.png)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 常用连续分布

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 正态分布
# MAGIC 
# MAGIC 若随机变量 ![](https://www.zhihu.com/equation?tex=X) 的密度函数为：
# MAGIC 
# MAGIC ![](https://gitee.com/wux_labs/Data-Scientist/raw/master/数学基础/概率论与数理统计/概率/20220303174330849.png)
# MAGIC 
# MAGIC 则称 ![](https://www.zhihu.com/equation?tex=X) 服从**正态分布**，称 ![](https://www.zhihu.com/equation?tex=X) 为**正态变量**。
# MAGIC 
# MAGIC 若随机变量 ![](https://www.zhihu.com/equation?tex=X) 服从一个**数学期望**为![](https://gitee.com/wux_labs/Data-Scientist/raw/master/数学基础/概率论与数理统计/概率/20220303175013621.png)、**方差**为![](https://gitee.com/wux_labs/Data-Scientist/raw/master/数学基础/概率论与数理统计/概率/20220303175053249.png)的正态分布，记为![](https://gitee.com/wux_labs/Data-Scientist/raw/master/数学基础/概率论与数理统计/概率/20220303175103605.png)。其中![](https://gitee.com/wux_labs/Data-Scientist/raw/master/数学基础/概率论与数理统计/概率/20220303175013621.png)为位置参数，用于控制曲线在 x 轴上的位置；![](https://gitee.com/wux_labs/Data-Scientist/raw/master/数学基础/概率论与数理统计/概率/20220303175120733.png)为尺度参数，用于控制曲线的形状。
# MAGIC 
# MAGIC 分布函数：
# MAGIC 
# MAGIC ![](https://gitee.com/wux_labs/Data-Scientist/raw/master/数学基础/概率论与数理统计/概率/20220303175331805.png)
# MAGIC 
# MAGIC > 数学期望：![](https://gitee.com/wux_labs/Data-Scientist/raw/master/数学基础/概率论与数理统计/概率/20220303175013621.png)  
# MAGIC 方差：![](https://gitee.com/wux_labs/Data-Scientist/raw/master/数学基础/概率论与数理统计/概率/20220303175053249.png)
# MAGIC 
# MAGIC **一些常用性质：**
# MAGIC 
# MAGIC 若 ![](https://gitee.com/wux_labs/Data-Scientist/raw/master/数学基础/概率论与数理统计/概率/20220303175103605.png) ：
# MAGIC 
# MAGIC ![](https://gitee.com/wux_labs/Data-Scientist/raw/master/数学基础/概率论与数理统计/概率/20220303182116418.png)
# MAGIC 
# MAGIC ![](https://gitee.com/wux_labs/Data-Scientist/raw/master/数学基础/概率论与数理统计/概率/20220303182129845.png)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 标准正态分布
# MAGIC 
# MAGIC 称 ![](https://gitee.com/wux_labs/Data-Scientist/raw/master/数学基础/概率论与数理统计/概率/20220303181305122.png) 时的正态分布为**标准正态分布**，记为 ![](https://gitee.com/wux_labs/Data-Scientist/raw/master/数学基础/概率论与数理统计/概率/20220303181709901.png)，其
# MAGIC 密度函数为：
# MAGIC 
# MAGIC ![](https://gitee.com/wux_labs/Data-Scientist/raw/master/数学基础/概率论与数理统计/概率/20220303181319249.png)
# MAGIC 
# MAGIC 分布函数为：
# MAGIC 
# MAGIC ![](https://gitee.com/wux_labs/Data-Scientist/raw/master/数学基础/概率论与数理统计/概率/20220303181328665.png)
# MAGIC 
# MAGIC 任何一个正态变量均可以通过**标准化**转化为**标准正态变量**，即若 ![](https://gitee.com/wux_labs/Data-Scientist/raw/master/数学基础/概率论与数理统计/概率/20220303175103605.png) ，则：
# MAGIC 
# MAGIC ![](https://gitee.com/wux_labs/Data-Scientist/raw/master/数学基础/概率论与数理统计/概率/20220303181517114.png)
# MAGIC 
# MAGIC 其中 ![](https://gitee.com/wux_labs/Data-Scientist/raw/master/数学基础/概率论与数理统计/概率/20220303181527149.png) 为**标准正态变量**。
# MAGIC 
# MAGIC 
# MAGIC **一些常用性质：**
# MAGIC 
# MAGIC 若 ![](https://gitee.com/wux_labs/Data-Scientist/raw/master/数学基础/概率论与数理统计/概率/20220303181709901.png) ：
# MAGIC 
# MAGIC ![](https://gitee.com/wux_labs/Data-Scientist/raw/master/数学基础/概率论与数理统计/概率/20220303182208846.png)
# MAGIC 
# MAGIC ![](https://gitee.com/wux_labs/Data-Scientist/raw/master/数学基础/概率论与数理统计/概率/20220303182221522.png)
# MAGIC 
# MAGIC ![](https://gitee.com/wux_labs/Data-Scientist/raw/master/数学基础/概率论与数理统计/概率/20220303182235230.png)
# MAGIC 
# MAGIC ![](https://gitee.com/wux_labs/Data-Scientist/raw/master/数学基础/概率论与数理统计/概率/20220303182249938.png)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 均匀分布
# MAGIC 
# MAGIC 若随机变量 ![](https://www.zhihu.com/equation?tex=X) 的密度函数为：
# MAGIC 
# MAGIC ![](https://gitee.com/wux_labs/Data-Scientist/raw/master/数学基础/概率论与数理统计/概率/20220303230957167.png)
# MAGIC 
# MAGIC 称 ![](https://www.zhihu.com/equation?tex=X) 服从区间 ![](https://gitee.com/wux_labs/Data-Scientist/raw/master/数学基础/概率论与数理统计/概率/20220303231014118.png) 上的**均匀分布**，记作 ![](https://gitee.com/wux_labs/Data-Scientist/raw/master/数学基础/概率论与数理统计/概率/20220303231028374.png) ，其分布函数：
# MAGIC 
# MAGIC ![](https://gitee.com/wux_labs/Data-Scientist/raw/master/数学基础/概率论与数理统计/概率/20220303231040295.png)
# MAGIC 
# MAGIC 均匀分布又称作**平顶分布**（因其概率密度为常值函数）。
# MAGIC 
# MAGIC > 数学期望： ![](https://gitee.com/wux_labs/Data-Scientist/raw/master/数学基础/概率论与数理统计/概率/20220303231052854.png)  
# MAGIC 方差： ![](https://gitee.com/wux_labs/Data-Scientist/raw/master/数学基础/概率论与数理统计/概率/20220303231104070.png) 

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 指数分布
# MAGIC 
# MAGIC 若随机变量 ![](https://www.zhihu.com/equation?tex=X) 的密度函数为：
# MAGIC 
# MAGIC ![](https://gitee.com/wux_labs/Data-Scientist/raw/master/数学基础/概率论与数理统计/概率/20220303232118227.png)
# MAGIC 
# MAGIC 称 ![](https://www.zhihu.com/equation?tex=X) 服从参数为 ![](https://gitee.com/wux_labs/Data-Scientist/raw/master/数学基础/概率论与数理统计/概率/20220303163125627.png) 的**指数分布**，记作 ![](https://gitee.com/wux_labs/Data-Scientist/raw/master/数学基础/概率论与数理统计/概率/20220303232130907.png) ，其分布函数：
# MAGIC 
# MAGIC ![](https://gitee.com/wux_labs/Data-Scientist/raw/master/数学基础/概率论与数理统计/概率/20220303232156627.png)
# MAGIC 
# MAGIC 指数分布是一种偏态分布，指数分布随机变量只可能取非负实数。指数分布常被用作各种“**寿命**”分布，譬如**电子元器件的寿命**、**动物的寿命**、**电话的通话时间**、**随机服务系统中的服务时间**等都可假定服从指数分布。指数分布在**可靠性**与**排队论**中有着广泛的应用。
# MAGIC 
# MAGIC > 数学期望： ![](https://gitee.com/wux_labs/Data-Scientist/raw/master/数学基础/概率论与数理统计/概率/20220303232210003.png)  
# MAGIC 方差： ![](https://gitee.com/wux_labs/Data-Scientist/raw/master/数学基础/概率论与数理统计/概率/20220303232219179.png) 

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 伽玛分布
# MAGIC 
# MAGIC **伽玛函数：**
# MAGIC 
# MAGIC ![](https://gitee.com/wux_labs/Data-Scientist/raw/master/数学基础/概率论与数理统计/概率/20220303234146870.png)
# MAGIC 
# MAGIC 其中参数 ![](https://gitee.com/wux_labs/Data-Scientist/raw/master/数学基础/概率论与数理统计/概率/20220303234206254.png) 。
# MAGIC 
# MAGIC 伽玛函数具有下列**性质：**
# MAGIC 
# MAGIC > ![](https://gitee.com/wux_labs/Data-Scientist/raw/master/数学基础/概率论与数理统计/概率/20220303234217142.png)  
# MAGIC ![](https://gitee.com/wux_labs/Data-Scientist/raw/master/数学基础/概率论与数理统计/概率/20220303234227366.png)
# MAGIC 
# MAGIC 当 ![](https://gitee.com/wux_labs/Data-Scientist/raw/master/数学基础/概率论与数理统计/概率/20220303234254654.png) 为自然数 n 时：
# MAGIC 
# MAGIC ![](https://gitee.com/wux_labs/Data-Scientist/raw/master/数学基础/概率论与数理统计/概率/20220303234316149.png)
# MAGIC 
# MAGIC **伽玛分布：**
# MAGIC 
# MAGIC 若随机变量 ![](https://www.zhihu.com/equation?tex=X) 的密度函数为：
# MAGIC 
# MAGIC ![](https://gitee.com/wux_labs/Data-Scientist/raw/master/数学基础/概率论与数理统计/概率/20220303234329757.png)
# MAGIC 
# MAGIC 称 ![](https://www.zhihu.com/equation?tex=X) 服从**伽玛分布**，记作 ![](https://gitee.com/wux_labs/Data-Scientist/raw/master/数学基础/概率论与数理统计/概率/20220303234359389.png) 。其中 ![](https://gitee.com/wux_labs/Data-Scientist/raw/master/数学基础/概率论与数理统计/概率/20220303234441037.png) 为**形状参数**， ![](https://gitee.com/wux_labs/Data-Scientist/raw/master/数学基础/概率论与数理统计/概率/20220303234455453.png) 为**尺度参数**。
# MAGIC 
# MAGIC > 数学期望： ![](https://gitee.com/wux_labs/Data-Scientist/raw/master/数学基础/概率论与数理统计/概率/20220303234506877.png)  
# MAGIC 方差： ![](https://gitee.com/wux_labs/Data-Scientist/raw/master/数学基础/概率论与数理统计/概率/20220303234516573.png) 

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 贝塔分布
# MAGIC 
# MAGIC **贝塔函数：**
# MAGIC 
# MAGIC ![](https://gitee.com/wux_labs/Data-Scientist/raw/master/数学基础/概率论与数理统计/概率/20220303235220307.png)
# MAGIC 
# MAGIC 其中参数 ![](https://gitee.com/wux_labs/Data-Scientist/raw/master/数学基础/概率论与数理统计/概率/20220303235232227.png) 。
# MAGIC 
# MAGIC 贝塔函数具有以下**性质：**
# MAGIC 
# MAGIC > 1、![](https://gitee.com/wux_labs/Data-Scientist/raw/master/数学基础/概率论与数理统计/概率/20220303235251468.png)  
# MAGIC 2、贝塔函数与伽玛函数有如下关系：![](https://gitee.com/wux_labs/Data-Scientist/raw/master/数学基础/概率论与数理统计/概率/20220303235303851.png)
# MAGIC 
# MAGIC **贝塔分布：**
# MAGIC 
# MAGIC 若随机变量 ![](https://www.zhihu.com/equation?tex=X) 的密度函数为：
# MAGIC 
# MAGIC ![](https://gitee.com/wux_labs/Data-Scientist/raw/master/数学基础/概率论与数理统计/概率/20220303235316788.png)
# MAGIC 
# MAGIC 则称 ![](https://www.zhihu.com/equation?tex=X) 服从**贝塔分布**，记作 ![](https://gitee.com/wux_labs/Data-Scientist/raw/master/数学基础/概率论与数理统计/概率/20220303235342595.png) ，其中 ![](https://gitee.com/wux_labs/Data-Scientist/raw/master/数学基础/概率论与数理统计/概率/20220303235232227.png) 都是**形状参数**。
# MAGIC 
# MAGIC > 数学期望： ![](https://gitee.com/wux_labs/Data-Scientist/raw/master/数学基础/概率论与数理统计/概率/20220303235357427.png)  
# MAGIC 方差： ![](https://gitee.com/wux_labs/Data-Scientist/raw/master/数学基础/概率论与数理统计/概率/20220303235406987.png) 

# COMMAND ----------

# MAGIC %md
# MAGIC # 条件概率
# MAGIC 
# MAGIC 假设我们知道给定事件B已经发生，在此基础上希望知道另一个事件A发生的可能性，此时我们就需要构造出条件概率，它需要先顾及事件B已经发生的信息，然后再求出事件A发生的概率。
# MAGIC 
# MAGIC **条件概率**描述的就是在给定事件B发生的情况下，事件A发生的概率，我们专门把他记作： ![](https://gitee.com/wux_labs/Data-Scientist/raw/master/数学基础/概率论与数理统计/概率/20220301164545310.png)。并且：
# MAGIC 
# MAGIC ![](https://gitee.com/wux_labs/Data-Scientist/raw/master/数学基础/概率论与数理统计/概率/20220301164707030.png)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 乘法公式
# MAGIC 
# MAGIC 根据条件概率的公式，我们可以得到如下等式：
# MAGIC 
# MAGIC ![](https://gitee.com/wux_labs/Data-Scientist/raw/master/数学基础/概率论与数理统计/概率/20220301164818839.png)

# COMMAND ----------

# MAGIC %md
# MAGIC # 全概率公式
# MAGIC 
# MAGIC 如果事件![](https://gitee.com/wux_labs/Data-Scientist/raw/master/数学基础/概率论与数理统计/概率/20220301165217654.png)满足：
# MAGIC 
# MAGIC * ![](https://gitee.com/wux_labs/Data-Scientist/raw/master/数学基础/概率论与数理统计/概率/20220301165217654.png) 两两互斥，也就是![](https://gitee.com/wux_labs/Data-Scientist/raw/master/数学基础/概率论与数理统计/概率/20220301165404179.png)，![](https://gitee.com/wux_labs/Data-Scientist/raw/master/数学基础/概率论与数理统计/概率/20220301165421610.png)，![](https://gitee.com/wux_labs/Data-Scientist/raw/master/数学基础/概率论与数理统计/概率/20220301165445749.png)，且![](https://gitee.com/wux_labs/Data-Scientist/raw/master/数学基础/概率论与数理统计/概率/20220301165617298.png)，![](https://www.zhihu.com/equation?tex=i%3D1%2C2%2C%5Ccdots)
# MAGIC * ![](https://gitee.com/wux_labs/Data-Scientist/raw/master/数学基础/概率论与数理统计/概率/20220301165806207.png)，则称事件组![](https://gitee.com/wux_labs/Data-Scientist/raw/master/数学基础/概率论与数理统计/概率/20220301165217654.png)是样本空间![](https://gitee.com/wux_labs/Data-Scientist/raw/master/数学基础/概率论与数理统计/概率/20220301165943919.png)的一个划分。设![](https://gitee.com/wux_labs/Data-Scientist/raw/master/数学基础/概率论与数理统计/概率/20220301165217654.png)是样本空间![](https://gitee.com/wux_labs/Data-Scientist/raw/master/数学基础/概率论与数理统计/概率/20220301165943919.png)的一个划分，A是任一事件。
# MAGIC 
# MAGIC 则：
# MAGIC 
# MAGIC ![](https://gitee.com/wux_labs/Data-Scientist/raw/master/数学基础/概率论与数理统计/概率/20220301170138359.png)

# COMMAND ----------

# MAGIC %md
# MAGIC # 贝叶斯公式
# MAGIC 
# MAGIC 与全概率公式解决的问题相反，贝叶斯公式是建立在条件概率的基础上寻找事件发生的原因（即大事件A已经发生的条件下，分割中的小事件![](https://gitee.com/wux_labs/Data-Scientist/raw/master/数学基础/概率论与数理统计/概率/20220301172315743.png)发生的概率）。
# MAGIC 
# MAGIC 设![](https://gitee.com/wux_labs/Data-Scientist/raw/master/数学基础/概率论与数理统计/概率/20220301165217654.png)是样本空间![](https://gitee.com/wux_labs/Data-Scientist/raw/master/数学基础/概率论与数理统计/概率/20220301165943919.png)的一个划分，则对任一事件A，![](https://gitee.com/wux_labs/Data-Scientist/raw/master/数学基础/概率论与数理统计/概率/20220301172555708.png)，有：
# MAGIC 
# MAGIC ![](https://gitee.com/wux_labs/Data-Scientist/raw/master/数学基础/概率论与数理统计/概率/20220301173113292.png)
# MAGIC 
# MAGIC ![](https://gitee.com/wux_labs/Data-Scientist/raw/master/数学基础/概率论与数理统计/概率/20220301172315743.png)常被视为导致试验结果A发生的“原因”，![](https://gitee.com/wux_labs/Data-Scientist/raw/master/数学基础/概率论与数理统计/概率/20220301173407060.png)，![](https://www.zhihu.com/equation?tex=i%3D1%2C2%2C%5Ccdots)，表示各种原因发生的可能性大小，故称**先验概率**。
# MAGIC 
# MAGIC ![](https://gitee.com/wux_labs/Data-Scientist/raw/master/数学基础/概率论与数理统计/概率/20220301173622768.png)，![](https://www.zhihu.com/equation?tex=i%3D1%2C2%2C%5Ccdots)，则反映当试验产生了结果A之后，再对各种原因概率的重新认识，故称**后验概率**。

# COMMAND ----------

# MAGIC %md
# MAGIC ## 朴素贝叶斯

# COMMAND ----------

# MAGIC %md
# MAGIC ### 朴素贝叶斯分类
# MAGIC 
# MAGIC ![](https://gitee.com/wux_labs/Data-Scientist/raw/master/数学基础/概率论与数理统计/概率/20220301175401692.png)

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
