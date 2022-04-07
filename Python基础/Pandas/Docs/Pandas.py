# Databricks notebook source
# MAGIC %md
# MAGIC 参考文档：http://www.pypandas.cn/docs/

# COMMAND ----------

# MAGIC %md
# MAGIC # Pandas 概览
# MAGIC Pandas 是 Python 的核心数据分析支持库，提供了快速、灵活、明确的数据结构，旨在简单、直观地处理关系型、标记型数据。Pandas 的目标是成为 Python 数据分析实践与实战的必备高级工具，其长远目标是成为最强大、最灵活、可以支持任何语言的开源数据分析工具。经过多年不懈的努力，Pandas 离这个目标已经越来越近了。
# MAGIC 
# MAGIC Pandas 适用于处理以下类型的数据：
# MAGIC 
# MAGIC * 与 SQL 或 Excel 表类似的，含异构列的表格数据;
# MAGIC * 有序和无序（非固定频率）的时间序列数据;
# MAGIC * 带行列标签的矩阵数据，包括同构或异构型数据;
# MAGIC * 任意其它形式的观测、统计数据集, 数据转入 Pandas 数据结构时不必事先标记。
# MAGIC 
# MAGIC Pandas 的主要数据结构是 Series（一维数据）与 DataFrame（二维数据），这两种数据结构足以处理金融、统计、社会科学、工程等领域里的大多数典型用例。对于 R 用户，DataFrame 提供了比 R 语言 data.frame 更丰富的功能。Pandas 基于 NumPy 开发，可以与其它第三方科学计算支持库完美集成。
# MAGIC 
# MAGIC Pandas 就像一把万能瑞士军刀，下面仅列出了它的部分优势 ：
# MAGIC 
# MAGIC * 处理浮点与非浮点数据里的缺失数据，表示为 NaN；
# MAGIC * 大小可变：插入或删除 DataFrame 等多维对象的列；
# MAGIC * 自动、显式数据对齐：显式地将对象与一组标签对齐，也可以忽略标签，在 Series、DataFrame 计算时自动与数据对齐；
# MAGIC * 强大、灵活的分组（group by）功能：拆分-应用-组合数据集，聚合、转换数据；
# MAGIC * 把 Python 和 NumPy 数据结构里不规则、不同索引的数据轻松地转换为 DataFrame 对象；
# MAGIC * 基于智能标签，对大型数据集进行切片、花式索引、子集分解等操作；
# MAGIC * 直观地合并（merge）、**连接（join）**数据集；
# MAGIC * 灵活地重塑（reshape）、**透视（pivot）**数据集；
# MAGIC * 轴支持结构化标签：一个刻度支持多个标签；
# MAGIC * 成熟的 IO 工具：读取文本文件（CSV 等支持分隔符的文件）、Excel 文件、数据库等来源的数据，利用超快的 HDF5 格式保存 / 加载数据；
# MAGIC * 时间序列：支持日期范围生成、频率转换、移动窗口统计、移动窗口线性回归、日期位移等时间序列功能。
# MAGIC 
# MAGIC 这些功能主要是为了解决其它编程语言、科研环境的痛点。处理数据一般分为几个阶段：数据整理与清洗、数据分析与建模、数据可视化与制表，Pandas 是处理数据的理想工具。
# MAGIC 
# MAGIC 其它说明：
# MAGIC 
# MAGIC * Pandas 速度很快。Pandas 的很多底层算法都用 Cython 优化过。然而，为了保持通用性，必然要牺牲一些性能，如果专注某一功能，完全可以开发出比 Pandas 更快的专用工具。
# MAGIC * Pandas 是 statsmodels 的依赖项，因此，Pandas 也是 Python 中统计计算生态系统的重要组成部分。
# MAGIC * Pandas 已广泛应用于金融领域。

# COMMAND ----------

# MAGIC %md
# MAGIC ## 数据结构
# MAGIC | 维数 | 名称 | 描述 |
# MAGIC | ------- | -------- | -------- |
# MAGIC | 1 | Series | 带标签的一维同构数组 |
# MAGIC | 2 | DataFrame | 带标签的，大小可变的，二维异构表格 |

# COMMAND ----------

# MAGIC %md
# MAGIC ### 为什么有多个数据结构？
# MAGIC 
# MAGIC Pandas 数据结构就像是低维数据的容器。比如，DataFrame 是 Series 的容器，Series 则是标量的容器。使用这种方式，可以在容器中以字典的形式插入或删除对象。
# MAGIC 
# MAGIC 此外，通用 API 函数的默认操作要顾及时间序列与截面数据集的方向。多维数组存储二维或三维数据时，编写函数要注意数据集的方向，这对用户来说是一种负担；如果不考虑 C 或 Fortran 中连续性对性能的影响，一般情况下，不同的轴在程序里其实没有什么区别。Pandas 里，轴的概念主要是为了给数据赋予更直观的语义，即用“更恰当”的方式表示数据集的方向。这样做可以让用户编写数据转换函数时，少费点脑子。
# MAGIC 
# MAGIC 处理 DataFrame 等表格数据时，index（行）或 columns（列）比 axis 0 和 axis 1 更直观。用这种方式迭代 DataFrame 的列，代码更易读易懂：
# MAGIC 
# MAGIC ```
# MAGIC for col in df.columns:
# MAGIC     series = df[col]
# MAGIC     # do something with series
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## 大小可变与数据复制
# MAGIC 
# MAGIC Pandas 所有数据结构的值都是可变的，但数据结构的大小并非都是可变的，比如，Series 的长度不可改变，但 DataFrame 里就可以插入列。
# MAGIC 
# MAGIC Pandas 里，绝大多数方法都不改变原始的输入数据，而是复制数据，生成新的对象。 一般来说，原始输入数据不变更稳妥。

# COMMAND ----------

# MAGIC %md
# MAGIC # Pandas 入门
# MAGIC 
# MAGIC 首先，导入 Pandas 与 NumPy。

# COMMAND ----------

import numpy as np
import pandas as pd

# COMMAND ----------

# MAGIC %md
# MAGIC ## Series 对象
# MAGIC 
# MAGIC Series是Pandas中最基本的对象，Series类似一种一维数组。事实上，Series基本上就是基于NumPy的数组对象来的。和NumPy的数组不同，**Series能为数据自定义标签，也就是索引（index）**，然后通过索引来访问数组中的数据。

# COMMAND ----------

# MAGIC %md
# MAGIC ### 生成对象

# COMMAND ----------

# MAGIC %md
# MAGIC #### 基于列表生成对象
# MAGIC 
# MAGIC 用值列表生成 Series 时，Pandas 默认自动生成整数索引：

# COMMAND ----------

import numpy as np
import pandas as pd

ser = pd.Series([1, 3, 5, np.nan, 6, 8])

print(ser)

# COMMAND ----------

# MAGIC %md
# MAGIC 通常我们会自己创建索引
# MAGIC > 注意：索引的个数与数据的个数应该相同

# COMMAND ----------

import numpy as np
import pandas as pd

ser1 = pd.Series([1, 3, 5, np.nan, 6, 8], index=['a','b','c','d','e','f'])
print(ser1)

ser2 = pd.Series([1, 3, 5, np.nan, 6, 8], index=list('ABCDEF'))
print(ser2)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 基于数据字典生成对象
# MAGIC 
# MAGIC 我们还可以将数据字典转换为Series
# MAGIC > 通过数据字典转换成Series，数据字典的Key将成为Series的Index。

# COMMAND ----------

import numpy as np
import pandas as pd

ser1 = pd.Series([1, 3, 5, np.nan, 6, 8], index=['a','b','c','d','e','f'])
print(ser1)

dict={'A':100,'B':200,'C':400}
ser2 = pd.Series(dict)
print(ser2)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 获取数据
# MAGIC 
# MAGIC Series对象提供了以下一些：
# MAGIC 
# MAGIC 属性：
# MAGIC * values 用来访问Series的值
# MAGIC * index 用来访问Series的索引
# MAGIC 
# MAGIC 方法：
# MAGIC * iteritems() 用来迭代访问数组中的对象

# COMMAND ----------

import numpy as np
import pandas as pd

ser0 = pd.Series([1, 3, 5, np.nan, 6, 8])
print("ser0的值：", ser0.values)
print("ser0的索引：", ser0.index)
print("ser0的键值对：", list(ser0.iteritems()))

ser1 = pd.Series([1, 3, 5, np.nan, 6, 8], index=['a','b','c','d','e','f'])
print("ser1的值：", ser1.values)
print("ser1的索引：", ser1.index)
print("ser1的键值对：", list(ser1.iteritems()))

ser2 = pd.Series([1, 3, 5, np.nan, 6, 8], index=list('ABCDEF'))
print("ser2的值：", ser2.values)
print("ser2的索引：", ser2.index)
print("ser2的键值对：", list(ser2.iteritems()))

# COMMAND ----------

# MAGIC %md
# MAGIC #### 索引下标获取
# MAGIC 
# MAGIC 对于Series的一维数组，我们可以通过索引下标来获取指定的元素

# COMMAND ----------

import numpy as np
import pandas as pd

ser1 = pd.Series([1, 3, 5, np.nan, 6, 8], index=['a','b','c','d','e','f'])

print(ser1)
print("通过索引'c'获取的数据：",ser1['c'])

# COMMAND ----------

# MAGIC %md
# MAGIC #### 位置下标获取
# MAGIC 
# MAGIC 对于Series的一维数组，我们可以通过位置下标来获取指定的元素

# COMMAND ----------

import numpy as np
import pandas as pd

ser1 = pd.Series([1, 3, 5, np.nan, 6, 8], index=['a','b','c','d','e','f'])

print(ser1)
print("通过位置 2 获取的数据：",ser1[2])

# COMMAND ----------

# MAGIC %md
# MAGIC #### 获取不连续的指定元素
# MAGIC 
# MAGIC 通过指定位置列表、索引列表，可以获取不连续的元素。

# COMMAND ----------

import numpy as np
import pandas as pd

ser1 = pd.Series([1, 3, 5, np.nan, 6, 8], index=['a','b','c','d','e','f'])

print(ser1)
print("通过索引获取不连续的元素：",ser1[['a','c','f']])
print("通过位置获取不连续的元素：",ser1[[0,2,5]])

# COMMAND ----------

# MAGIC %md
# MAGIC #### 使用切片获取
# MAGIC 
# MAGIC 不仅可以通过位置进行切片，还可以通过索引进行切片，指定起始索引名称及结束索引名称即可。

# COMMAND ----------

import numpy as np
import pandas as pd

ser1 = pd.Series([1, 3, 5, np.nan, 6, 8], index=['a','b','c','d','e','f'])

print(ser1)
print("通过索引切片的元素：",ser1['b':'e'])
print("通过位置切片的元素：",ser1[1:5])

# COMMAND ----------

# MAGIC %md
# MAGIC ### 索引操作

# COMMAND ----------

# MAGIC %md
# MAGIC #### 重建索引
# MAGIC 
# MAGIC 当一个Series创建好后，可以通过对 index 赋值来改变原有的索引信息。

# COMMAND ----------

import numpy as np
import pandas as pd

ser1 = pd.Series([1, 3, 5, np.nan, 6, 8], index=['a','b','c','d','e','f'])

print(ser1)

ser1.index = list('fedcba')

print(ser1)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 重新索引
# MAGIC 
# MAGIC reindex()表示重新索引，不会改变原始Series的内容，而是返回一个新的Series对象。
# MAGIC * 如果某个索引值当前不存在，就会引入缺失值；
# MAGIC * 如果某个索引值当前存在，就会按新索引的顺序调整值的顺序。

# COMMAND ----------

import numpy as np
import pandas as pd

ser1 = pd.Series([1, 3, 5, np.nan, 6, 8], index=['a','b','c','d','e','f'])

print(ser1)

# 根据新的索引重新排序Series的值
ser2 = ser1.reindex(list('fedcba'))
# 原始索引不存在，所以会引入缺失值NaN
ser3 = ser1.reindex(list('efghij'))

# 原始Series不会变
print(ser1)
print(ser2)
print(ser3)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 删除数据
# MAGIC 
# MAGIC 可以通过drop()方法来删除指定的数据。

# COMMAND ----------

import numpy as np
import pandas as pd

ser1 = pd.Series([1, 3, 5, np.nan, 6, 8], index=['a','b','c','d','e','f'])

ser2 = ser1.drop(['b','d','f'])

print(ser1)
print(ser2)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 数据运算

# COMMAND ----------

# MAGIC %md
# MAGIC #### 算术运算
# MAGIC 
# MAGIC 两个Series进行算数运算，会根据索引进行匹配进行运算：
# MAGIC * 如果一个索引在两个Series中同时存在，则对应元素的值进行运算；
# MAGIC * 如果一个索引只存在于一个Series中，则保留索引，并引入确实值。

# COMMAND ----------

import numpy as np
import pandas as pd

ser1 = pd.Series([1, 3, 5, 4, 6, 8], index=['a','b','c','d','e','f'])
ser2 = pd.Series([2, 4, 6, 5, 7, 9], index=['d','e','f','h','i','j'])

print(ser1)
print(ser2)

# 两个Series对应索引的元素进行运算，找不到的索引引入缺失值
print(ser1 + ser2)
print(ser1 - ser2)
print(ser1 * ser2)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 其他运算
# MAGIC 
# MAGIC Series还支持类似于NumPy数组的一些其他运算，比如：
# MAGIC * 数组与数的运算
# MAGIC * 布尔过滤
# MAGIC * 数学函数
# MAGIC * ... ...

# COMMAND ----------

import numpy as np
import pandas as pd

ser1 = pd.Series([1, 3, 5, 4, 6, 8], index=['a','b','c','d','e','f'])
ser2 = pd.Series([2, 4, 6, 5, 7, 9], index=['d','e','f','h','i','j'])

print(ser1)

# 布尔过滤
print(ser1[ser1>4])
# 数组与数的运算
print(ser1 * 2)
# 数学函数运算
print(np.square(ser1))

# COMMAND ----------

# MAGIC %md
# MAGIC ## DataFrame 对象
# MAGIC 
# MAGIC DataFrame是一个二维的表结构，Pandas的DataFrame可以存储许多种不同的数据类型，并且每一个坐标轴都有自己的标签，可以把他想象成一个Series的字典项。
# MAGIC 
# MAGIC DataFrame的三个参数data、index和columns分别为数据、行索引和列索引。

# COMMAND ----------

# MAGIC %md
# MAGIC ### 生成对象

# COMMAND ----------

# MAGIC %md
# MAGIC #### 基于列表生成对象
# MAGIC 
# MAGIC 用值列表生成 DataFrame 时，Pandas 默认自动生成整数索引：

# COMMAND ----------

import numpy as np
import pandas as pd

df = pd.DataFrame(np.random.randint(0,10,(5,5)))

print(df)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC 在创建DataFrame的时候，可以指定自定义的索引。

# COMMAND ----------

import numpy as np
import pandas as pd

df1 = pd.DataFrame(np.random.randint(0,10,(5,5)), index=['i1','i2','i3','i4','i5'])

df2 = pd.DataFrame(np.random.randint(0,10,(5,5)), columns=['c1','c2','c3','c4','c5'])

df3 = pd.DataFrame(np.random.randint(0,10,(5,5)), index=['i1','i2','i3','i4','i5'], columns=['c1','c2','c3','c4','c5'])

print(df1)
print(df2)
print(df3)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 基于数据字典生成对象
# MAGIC 
# MAGIC 我们还可以将一个数据字典转换成DataFrame。
# MAGIC * 可以直接将数据字典作为DataFrame的参数进行构造；
# MAGIC * 也可以通过from_dict()函数进行构造。

# COMMAND ----------

import numpy as np
import pandas as pd

df1 = pd.DataFrame(np.random.randint(0,10,(5,5)), index=['i1','i2','i3','i4','i5'], columns=['c1','c2','c3','c4','c5'])


dict1 = {"a1":{"c1":11,"c2":22,"c3":33},"a2":{"c1":21,"c2":32,"c3":43},"a3":{"c1":41,"c2":52,"c3":63}}
df2 = pd.DataFrame(dict1)

print(df1)
print(df2)

dict2 = {"prov":["beijing","shanghai","shenzhen","guangzhou"],
        "prop":[1.1,2.2,3.3,4.4],
        "year":[2018,2019,2020,2021]}
df3 = pd.DataFrame(dict2)
df4 = pd.DataFrame.from_dict(dict2)
print(df3)
print(df4)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 基于Series生成对象
# MAGIC 
# MAGIC 我们还可以基于Series对象生成DataFrame对象。
# MAGIC 
# MAGIC > 由于每个Series的索引可能不一样，遇到不存在的索引时会引入缺失值。

# COMMAND ----------

import numpy as np
import pandas as pd

df1 = pd.DataFrame(np.random.randint(0,10,(5,5)), index=['i1','i2','i3','i4','i5'], columns=['c1','c2','c3','c4','c5'])


dict1 = {"a1":pd.Series([11,22,33],index=['c1','c2','c3']),"a2":pd.Series([21,32,43],index=['c1','c2','c3']),"a3":pd.Series([31,42,53],index=['c1','c2','c3'])}
df2 = pd.DataFrame(dict1)

dict2 = {"a1":pd.Series([11,22,33],index=['c1','c2','c3']),"a2":pd.Series([21,32,43],index=['c1','c2','c3']),"a3":pd.Series([31,42,53],index=['c1','c2','c3']),"a4":pd.Series([31,42,53],index=['c1','c2','c4'])}
df3 = pd.DataFrame(dict2)
print(df1)
print(df2)
print(df3)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 常用的属性
# MAGIC 
# MAGIC DataFrame提供了一些常用的
# MAGIC 
# MAGIC 属性：
# MAGIC 
# MAGIC * shape 获取DataFrame的形状
# MAGIC * index 获取DataFrame的行索引
# MAGIC * columns 获取DataFrame的列索引
# MAGIC * dtypes 获取DataFrame的数据类型
# MAGIC * ndim 获取维度
# MAGIC 
# MAGIC 方法：
# MAGIC 
# MAGIC * info() 返回DataFrame的信息

# COMMAND ----------

import numpy as np
import pandas as pd

dict = {"prov":["beijing","shanghai","shenzhen","guangzhou"],
        "prop":[1.1,2.2,3.3,4.4],
        "year":[2018,2019,2020,2021]}
df = pd.DataFrame(dict)
print(df)

print("形状：", df.shape)
print("行索引：", df.index, list(df.index))
print("列索引：", df.columns, df.columns.tolist())
print("数据类型：", df.dtypes)
print("数据维度：", df.ndim)

print("DataFrame的信息：")
df.info()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 获取数据

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 数据抽样
# MAGIC 
# MAGIC * head() 获取前几行的数据，默认获取5行
# MAGIC * tail() 获取后几行的数据，默认获取5行

# COMMAND ----------

import numpy as np
import pandas as pd

df = pd.DataFrame(np.random.randint(0,10,(100,5)))
print(df)

# 获取前几行
print("获取前几行：\n", df.head())
print("获取前8行：\n", df.head(8))
# 获取后几行
print("获取后几行：\n", df.tail())
print("获取后8行：\n", df.tail(8))

# COMMAND ----------

# MAGIC %md
# MAGIC #### 索引下标获取
# MAGIC 
# MAGIC 我们可以通过DataFrame的索引获取数据
# MAGIC * 当获取指定一列，返回的数据是Series
# MAGIC * 当获取指定多列，返回的数据是DataFrame

# COMMAND ----------

import numpy as np
import pandas as pd

dict = {"prov":["beijing","shanghai","shenzhen","guangzhou"],
        "prop":[1.1,2.2,3.3,4.4],
        "year":[2018,2019,2020,2021]}
df = pd.DataFrame(dict)
print(df)

# 获取DataFrame的列
print("获取一列：", df["year"], type(df["year"]))

# 指定索引列表，可以获取多列
print("获取多列：", df[["year","prov"]], type(df[["year","prov"]]))


# COMMAND ----------

# MAGIC %md
# MAGIC 我们也可以通过行索引切片获取DataFrame的行数据
# MAGIC * 当获取指定一行，返回的数据是DataFrame
# MAGIC * 当获取指定多行，返回的数据是DataFrame

# COMMAND ----------

import numpy as np
import pandas as pd

dict = {"prov":["beijing","shanghai","shenzhen","guangzhou"],
        "prop":[1.1,2.2,3.3,4.4],
        "year":[2018,2019,2020,2021]}
df = pd.DataFrame(dict)
print(df)

# 获取DataFrame的行
print("获取一行：", df[1:2], type(df[1:2]))

# 指定索引切片，可以获取多行
print("获取多行：", df[1:3], type(df[1:3]))

# COMMAND ----------

# MAGIC %md
# MAGIC 结合行索引切片和列索引，可以取多行多列数据

# COMMAND ----------

import numpy as np
import pandas as pd

dict = {"prov":["beijing","shanghai","shenzhen","guangzhou"],
        "prop":[1.1,2.2,3.3,4.4],
        "year":[2018,2019,2020,2021]}
df = pd.DataFrame(dict)
print(df)

print("获取一行一列：", df[1:2]["year"], type(df[1:2]["year"]))
print("获取一行多列：", df[1:2][["year","prov"]], type(df[1:2][["year","prov"]]))
print("获取多行一列：", df[1:3]["year"], type(df[1:3]["year"]))
print("获取多行多列：", df[1:3][["year","prov"]], type(df[1:3][["year","prov"]]))

# COMMAND ----------

# MAGIC %md
# MAGIC #### 通过方法获取
# MAGIC 
# MAGIC DataFrame提供了：
# MAGIC * loc 通过索引进行取数据
# MAGIC * iloc 通过位置进行取数据

# COMMAND ----------

import numpy as np
import pandas as pd

dict = {"prov":["beijing","shanghai","shenzhen","guangzhou"],
        "prop":[1.1,2.2,3.3,4.4],
        "year":[2018,2019,2020,2021]}
df = pd.DataFrame(dict, index=['bj','sh','sz','gz'])
print(df)

print("获取一行一列：", df.loc['sh', "year"], type(df.loc['sh', "year"]))
print("获取一行多列：", df.loc['sh', ["year","prov"]], type(df.loc['sh', ["year","prov"]]))
print("获取一行多列：", df.loc['sh', :], type(df.loc['sh', :]))
print("获取多行一列：", df.loc[['sh','gz'], ["year"]], type(df.loc[['sh','gz'], ["year"]]))
print("获取多行多列：", df.loc[['sh','gz'], ["year","prov"]], type(df.loc[['sh','gz'], ["year","prov"]]))

print("获取一行：", df.iloc[1], type(df.iloc[1]))
print("获取连续多行：", df.iloc[1:3], type(df.iloc[1:3]))
print("获取不连续多行：", df.iloc[[1,3]], type(df.iloc[[1,3]]))

print("获取一行一列：", df.iloc[1,2], type(df.iloc[1,2]))
print("获取多行多列：", df.iloc[[1,2],[0,2]], type(df.iloc[[1,2],[0,2]]))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 索引操作

# COMMAND ----------

# MAGIC %md
# MAGIC #### 修改index
# MAGIC 
# MAGIC 当一个DataFrame创建好后，可以通过对 index 赋值来改变原有的行索引信息。

# COMMAND ----------

import numpy as np
import pandas as pd

df = pd.DataFrame(np.random.randint(0,10,(5,5)), index=['i1','i2','i3','i4','i5'], columns=['c1','c2','c3','c4','c5'])

print(df)

print(df.index)

df.index = ['index1','index2','index3','index4','index5']

print(df)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 修改columns
# MAGIC 
# MAGIC 当一个DataFrame创建好后，可以通过对 columns 赋值来改变原有的列索引信息。

# COMMAND ----------

import numpy as np
import pandas as pd

df = pd.DataFrame(np.random.randint(0,10,(5,5)), index=['i1','i2','i3','i4','i5'], columns=['c1','c2','c3','c4','c5'])

print(df)

print(df.columns)

df.columns = ['column1','column2','column3','column4','column5']

print(df)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 通过方法修改索引
# MAGIC 
# MAGIC DataFrame提供了rename()方法来修改索引名称。
# MAGIC * 参数index指定修改行索引
# MAGIC * 参数columns执行修改列索引

# COMMAND ----------

import numpy as np
import pandas as pd

df1 = pd.DataFrame(np.random.randint(0,10,(5,5)), index=['i1','i2','i3','i4','i5'], columns=['c1','c2','c3','c4','c5'])

print(df1)

def getName(x):
    return x + "_renamed"

# 修改行索引
df2 = df1.rename(index=getName)
print(df2)

# 修改列索引
df3 = df1.rename(columns=getName)
print(df3)

# 同时修改行列索引
df4 = df1.rename(index=getName,columns=getName)
print(df4)

# 只修改部分行列的索引
df5 = df1.rename(index={"i2":"i2_renamed"},columns={"c1":"c1_renamed","c3":"c3_renamed"})
print(df5)

print(df1)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 使用某一列作为索引
# MAGIC 
# MAGIC DataFrame提供了set_index()方法，可以将一列数据设置为行索引。
# MAGIC 
# MAGIC * 参数drop 用来指定某一列数据被作为行索引后是否删除原始数据，默认是True

# COMMAND ----------

import numpy as np
import pandas as pd

dict = {"prov":["beijing","shanghai","shenzhen","guangzhou"],
        "prop":[1.1,2.2,3.3,4.4],
        "year":[2018,2019,2020,2021]}
df1 = pd.DataFrame(dict)
print(df1)

df2 = df1.set_index("prov")
print(df2)

df3 = df1.set_index("prov", drop=False)
print(df3)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 数据操作

# COMMAND ----------

# MAGIC %md
# MAGIC #### 添加数据
# MAGIC 
# MAGIC 通过直接赋值的方式可以增加数据。

# COMMAND ----------

import numpy as np
import pandas as pd

dict = {"prov":["beijing","shanghai","shenzhen","guangzhou"],
        "prop":[1.1,2.2,3.3,4.4],
        "year":[2018,2019,2020,2021]}
df1 = pd.DataFrame(dict)
print(df1)

# 直接对列进行赋值
df1["GDP"] = [100,100,100,100]
print(df1)

# 先添加列
# 再对列进行赋值
col_names = df1.columns.tolist()
col_names.insert(2,"location")
df2 = df1.reindex(columns=col_names)
print(df2)
df2["location"] = [101,101,101,101]
print(df2)
