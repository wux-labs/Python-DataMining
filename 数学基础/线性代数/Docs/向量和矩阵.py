# Databricks notebook source
# MAGIC %md
# MAGIC # 向量

# COMMAND ----------

# MAGIC %md
# MAGIC ## 什么是向量
# MAGIC 
# MAGIC **向量是指具有大小和方向的量**，它可以形象化地表示为带箭头的线段。箭头所指：代表向量的方向；线段长度：代表向量的大小。
# MAGIC 
# MAGIC 一般记做： ![](https://www.zhihu.com/equation?tex=a) ， ![](https://www.zhihu.com/equation?tex=%5Cvec%7Ba%7D)，![](https://www.zhihu.com/equation?tex=%5Cvec%7BAB%7D)，同时也可以用数对的形式表示，例如：![](https://www.zhihu.com/equation?tex=%5Cleft+%28+x%2Cy+%5Cright+%29) ，![](https://www.zhihu.com/equation?tex=%5Cleft+%28+7%2C8%2C9+%5Cright+%29)
# MAGIC 
# MAGIC **向量的矩阵**表示：
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=%5Cvec%7Ba%7D%3D%5Cbegin%7Bbmatrix%7Dx%5C%5Cy%5Cend%7Bbmatrix%7D) 
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=%5Cvec%7Ba%7D%3D%5Cbegin%7Bbmatrix%7Dx%5C%5Cy%5C%5Cz%5Cend%7Bbmatrix%7D) 
# MAGIC 
# MAGIC **向量的大小**，也就是向量的长度（一般称作为 模），向量![](https://www.zhihu.com/equation?tex=%5Cvec%7Ba%7D)的模记为： ![](https://www.zhihu.com/equation?tex=%5Cleft+%7C+%5Cvec%7Ba%7D+%5Cright+%7C) ，若 ![](https://www.zhihu.com/equation?tex=%5Cvec%7Ba%7D+%3D+%5Cleft+%28+x%2Cy%2Cz+%5Cright+%29) ，则 ![](https://www.zhihu.com/equation?tex=%5Cleft+%7C+%5Cvec%7Ba%7D+%5Cright+%7C+%3D+%5Csqrt%7Bx%5E2+%2B+y%5E2%2B+z%5E2%7D)
# MAGIC 
# MAGIC **单位向量**：即模为1的向量，可以记作 ![](https://www.zhihu.com/equation?tex=%5Chat%7Ba%7D) 。一个向量的单位向量，可以通过除以它模得到，即 ![](https://www.zhihu.com/equation?tex=%5Chat%7Ba%7D+%3D+%5Cfrac%7B%5Cvec%7Ba%7D%7D%7B%7C+%5Cvec%7Ba%7D%7C%7D) 。
# MAGIC 
# MAGIC **零向量**：即模为0的向量，零向量的方向是任意的
# MAGIC 
# MAGIC **相反向量**：长度相等方向相反的向量， ![](https://www.zhihu.com/equation?tex=%5Cvec%7Ba%7D) 的相反向量为 ![](https://www.zhihu.com/equation?tex=-%5Cvec%7Ba%7D)
# MAGIC 
# MAGIC **平行（共线）向量**：方向相同或相反的非零向量，记作 ![](https://www.zhihu.com/equation?tex=%5Cvec%7Ba%7D+%2F%2F+%5Cvec%7Bb%7D)

# COMMAND ----------

import numpy as np

np.array((7,8,9))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 向量的运算
# MAGIC 
# MAGIC 设 ![](https://www.zhihu.com/equation?tex=%5Cvec%7Ba%7D%3D%28x_1%2Cy_1%2Cz_1%29) ， ![](https://www.zhihu.com/equation?tex=%5Cvec%7Bb%7D%3D%28x_2%2Cy_2%2Cz_2%29)。

# COMMAND ----------

# MAGIC %md
# MAGIC ### 向量的加法
# MAGIC 
# MAGIC 向量的加法满足平行四边形法则和三角形法则，![](https://www.zhihu.com/equation?tex=%5Cvec%7BOA%7D+%2B+%5Cvec%7BOB%7D+%3D+%5Cvec%7BOC%7D)。
# MAGIC 
# MAGIC 将向量中对应位置的元素相加即可。
# MAGIC 
# MAGIC **运算过程：**
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=%5Cvec%7Ba%7D+%2B+%5Cvec%7Bb%7D+%3D+%28x_1%2Cy_1%2Cz_1%29%2B%28x_2%2Cy_2%2Cz_2%29%3D%28x_1%2Bx_2%2Cy_1%2By_2%2Cz_1%2Bz_2%29)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=%5Cvec%7Ba%7D+%2B+%5Cvec%7Bb%7D+%3D+%5Cbegin%7Bbmatrix%7D+x_%7B1%7D%5C%5C+y_%7B1%7D+%5C%5Cz_1%5Cend%7Bbmatrix%7D+%2B+%5Cbegin%7Bbmatrix%7D+x_%7B2%7D%5C%5C+y_%7B2%7D+%5C%5Cz_2%5Cend%7Bbmatrix%7D+%3D+%5Cbegin%7Bbmatrix%7D+x_%7B1%7D+%2B+x_%7B2%7D%5C%5C+y_%7B1%7D+%2B+y_%7B2%7D+%5C%5Cz_1%2Bz_2%5Cend%7Bbmatrix%7D)

# COMMAND ----------

import numpy as np

a = np.array((7,8,9))
b = np.array((4,5,6))

print(a + b)

# COMMAND ----------

# MAGIC %md
# MAGIC **一些运算律**
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=%5Cvec%7Ba%7D+%2B+0+%3D+0+%2B+%5Cvec%7Ba%7D+%3D+%5Cvec%7Ba%7D)
# MAGIC 
# MAGIC **交换律：**![](https://www.zhihu.com/equation?tex=%5Cvec%7Ba%7D+%2B+%5Cvec%7Bb%7D+%3D+%5Cvec%7Bb%7D+%2B+%5Cvec%7Ba%7D)
# MAGIC 
# MAGIC **结合律：**![](https://www.zhihu.com/equation?tex=%5Cleft+%28+%5Cvec%7Ba%7D+%2B+%5Cvec%7Bb%7D+%5Cright%29+%2B+%5Cvec%7Bc%7D+%3D+%5Cvec%7Ba%7D+%2B+%5Cleft+%28+%5Cvec%7Bb%7D+%2B+%5Cvec%7Bc%7D+%5Cright%29)

# COMMAND ----------

import numpy as np

a = np.array((7,8,9))
b = np.array((4,5,6))
c = np.array((1,2,3))

print("交换律")
print(a + b)
print(b + a)

print("结合律")
print(a + b + c)
print((a + b) + c)
print(a + (b + c))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 向量的减法
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=%5Cvec%7BOA%7D+-+%5Cvec%7BOB%7D+%3D+%5Cvec%7BBA%7D)。
# MAGIC 
# MAGIC 将向量中对应位置的元素相减即可。
# MAGIC 
# MAGIC **运算过程：**
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=%5Cvec%7Ba%7D+-+%5Cvec%7Bb%7D+%3D+%28x_1%2Cy_1%2Cz_1%29-%28x_2%2Cy_2%2Cz_2%29%3D%28x_1-x_2%2Cy_1-y_2%2Cz_1-z_2%29)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=%5Cvec%7Ba%7D+-+%5Cvec%7Bb%7D+%3D+%5Cbegin%7Bbmatrix%7D+x_%7B1%7D%5C%5C+y_%7B1%7D+%5C%5Cz_1%5Cend%7Bbmatrix%7D+-+%5Cbegin%7Bbmatrix%7D+x_%7B2%7D%5C%5C+y_%7B2%7D%5C%5Cz_2+%5Cend%7Bbmatrix%7D+%3D+%5Cbegin%7Bbmatrix%7D+x_%7B1%7D+-+x_%7B2%7D%5C%5C+y_%7B1%7D+-+y_%7B2%7D+%5C%5Cz_1-z_2%5Cend%7Bbmatrix%7D)

# COMMAND ----------

import numpy as np

a = np.array((7,8,9))
b = np.array((4,5,6))

print(a - b)

# COMMAND ----------

# MAGIC %md
# MAGIC **一些运算律**
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=%5Cvec%7Ba%7D%2B%28-%5Cvec%7Bb%7D%29%3D%5Cvec%7Ba%7D-%5Cvec%7Bb%7D)

# COMMAND ----------

import numpy as np

a = np.array((7,8,9))
b = np.array((4,5,6))

print(a + (-b))
print(a - b)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 实数和向量的积
# MAGIC 
# MAGIC 设有实数![](https://www.zhihu.com/equation?tex=k)，和向量 ![](https://www.zhihu.com/equation?tex=%5Cvec%7Ba%7D) 的乘积还是一个向量，记做 ![](https://www.zhihu.com/equation?tex=k%5Cvec%7Ba%7D) ，且 ![](https://www.zhihu.com/equation?tex=+%7C+k+%5Cvec%7Ba%7D+%7C+%3D++%7C+k%7C+%2A+%7C+%5Cvec%7Ba%7D+%7C) ，如果 ![](https://www.zhihu.com/equation?tex=k%5Cvec%7Ba%7D%3D0) ，则 ![](https://www.zhihu.com/equation?tex=k%3D0) 或 ![](https://www.zhihu.com/equation?tex=%5Cvec%7Ba%7D%3D0)
# MAGIC 
# MAGIC 其几何意义为：向量的有向线段的伸长或者压缩。

# COMMAND ----------

import numpy as np

a = np.array((7,8,9))

print(a * 3)
print(a * 1/3)
print(a * 0)

# COMMAND ----------

# MAGIC %md
# MAGIC **一些运算律：**
# MAGIC 
# MAGIC **结合律：** ![](https://www.zhihu.com/equation?tex=%28+k%5Cvec%7Ba%7D+%29%5Ccdot+%5Cvec%7Bb%7D+%3D+k%28%5Cvec%7Ba%7D+%5Ccdot+%5Cvec%7Bb%7D+%29+%3D+%5Cvec%7Ba%7D+%5Ccdot+%28+k%5Cvec%7Bb%7D+%29)
# MAGIC 
# MAGIC **分配律：** ![](https://www.zhihu.com/equation?tex=%28+j%2B+k%29+%5Cvec%7Ba%7D+%3D+j%5Cvec%7Ba%7D+%2B+k%5Cvec%7Ba%7D)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=k%28+%5Cvec%7Ba%7D+%2B+%5Cvec%7Bb%7D+%29+%3Dk%5Cvec%7Ba%7D+%2Bk%5Cvec%7Bb%7D)

# COMMAND ----------

import numpy as np

a = np.array((7,8,9))
b = np.array((4,5,6))

print("结合律")
print(3 * a * b)
print((3 * a) * b)
print(3 * (a * b))

print("分配律")
print((3 + 2) * a)
print(3 * a + 2 * a)

print(3 * (a + b))
print(3 * a + 3 * b)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 向量的点乘（点积，内积，数量积）
# MAGIC 
# MAGIC 两个向量的数量积（点积，内积，点乘）是一个数量，没有方向，记作 ![](https://www.zhihu.com/equation?tex=%5Cvec%7Ba%7D%5Ccdot+%5Cvec%7Bb%7D)
# MAGIC 
# MAGIC **代数定义：** ![](https://www.zhihu.com/equation?tex=%5Cvec%7Ba%7D%5Ccdot+%5Cvec%7Bb%7D+%3D+x_%7B1%7Dx_%7B2%7D%2By_%7B1%7Dy_%7B2%7D%2Bz_1z_2)
# MAGIC 
# MAGIC **几何定义：** 我们将 ![](https://www.zhihu.com/equation?tex=%5Cvec%7Ba%7D) 和 ![](https://www.zhihu.com/equation?tex=%5Cvec%7Bb%7D) 的夹角记作 ![](https://www.zhihu.com/equation?tex=%5Ctheta) ，且 ![](https://www.zhihu.com/equation?tex=0%5Cleqslant+%5Ctheta+%5Cleqslant+%5Cpi)
# MAGIC 
# MAGIC 若 ![](https://www.zhihu.com/equation?tex=%5Cvec%7Ba%7D) ， ![](https://www.zhihu.com/equation?tex=%5Cvec%7Bb%7D) 不共线，则 ![](https://www.zhihu.com/equation?tex=%5Cvec%7Ba%7D%5Ccdot+%5Cvec%7Bb%7D+%3D+%7C+%5Cvec%7Ba%7D+%7C+%5Ccdot+%7C+%5Cvec%7Bb%7D+%7C%5Ccdot+cos%5Ctheta)
# MAGIC 
# MAGIC 若 ![](https://www.zhihu.com/equation?tex=%5Cvec%7Ba%7D) ， ![](https://www.zhihu.com/equation?tex=%5Cvec%7Bb%7D) 共线，则 ![](https://www.zhihu.com/equation?tex=%5Cvec%7Ba%7D%5Ccdot+%5Cvec%7Bb%7D+%3D+%5Cpm+%5Cleft+%7C+%5Cvec%7Ba%7D+%5Cright+%7C+%5Ccdot+%5Cleft+%7C+%5Cvec%7Bb%7D+%5Cright+%7C) ，因为此时 ![](https://www.zhihu.com/equation?tex=%5Ctheta%3D0) 则 ![](https://www.zhihu.com/equation?tex=%5Ccos%5Ctheta%3D1)，若两个向量方向相反，则认为![](https://www.zhihu.com/equation?tex=%5Ctheta%3D%5Cpi) 则 ![](https://www.zhihu.com/equation?tex=%5Ccos%5Ctheta%3D-1)。

# COMMAND ----------

import numpy as np

a = np.array((7,8,9))
b = np.array((4,5,6))

print(a.dot(b))

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC **一些运算律：**
# MAGIC 
# MAGIC **交换律：** ![](https://www.zhihu.com/equation?tex=%5Cvec%7Ba%7D%5Ccdot+%5Cvec%7Bb%7D%3D%5Cvec%7Bb%7D%5Ccdot+%5Cvec%7Ba%7D)
# MAGIC 
# MAGIC **结合律：** ![](https://www.zhihu.com/equation?tex=%28+k%5Cvec%7Ba%7D+%29%5Ccdot+%5Cvec%7Bb%7D+%3D+k%28+%5Cvec%7Ba%7D%5Ccdot+%5Cvec%7Bb%7D+%29)
# MAGIC 
# MAGIC **分配率：** ![](https://www.zhihu.com/equation?tex=%28+%5Cvec%7Ba%7D+%2B+%5Cvec%7Bb%7D+%29%5Ccdot+%5Cvec%7Bc%7D+%3D%5Cvec%7Ba%7D%5Ccdot+%5Cvec%7Bc%7D+%2B+%5Cvec%7Bb%7D%5Ccdot+%5Cvec%7Bc%7D)
# MAGIC 
# MAGIC **一些性质：**
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=%5Cvec%7Ba%7D%5Ccdot+%5Cvec%7Ba%7D+%3D+%7C+%5Cvec%7Ba%7D+%7C+%5E%7B2%7D)
# MAGIC 
# MAGIC 若两个向量互相垂直，则 ![](https://www.zhihu.com/equation?tex=%5Ccos%5Ctheta%3D0) 因此 ![](https://www.zhihu.com/equation?tex=%5Cvec%7Ba%7D%5Cperp+%5Cvec%7Bb%7D+%5Cleftrightharpoons+%5Cvec%7Ba%7D%5Ccdot+%5Cvec%7Bb%7D+%3D+0)

# COMMAND ----------

import numpy as np

a = np.array((7,8,9))
b = np.array((4,5,6))
c = np.array((1,2,3))

print("交换律")
print(a.dot(b))
print(b.dot(a))

print("结合律")
print((5 * b).dot(a))
print(5 * (b.dot(a)))

print("分配律")
print((a + b).dot(c))
print(a.dot(c) + b.dot(c))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 向量的叉乘（叉积，外积，向量积）
# MAGIC 
# MAGIC 两个向量的向量积（叉积，叉乘，外积）是一个向量，记作 ![](https://www.zhihu.com/equation?tex=%5Cvec%7Ba%7D%5Ctimes+%5Cvec%7Bb%7D) （或者 ![](https://www.zhihu.com/equation?tex=%5Cvec%7Ba%7D+%5Cwedge+%5Cvec%7Bb%7D)）
# MAGIC 
# MAGIC **代数定义：** ![](https://www.zhihu.com/equation?tex=%5Cvec%7Ba%7D%5Ctimes+%5Cvec%7Bb%7D%3D+%5Cleft+%28+y_%7B1%7Dz_%7B2%7D-z_%7B1%7Dy_%7B2%7D+%2Cz_%7B1%7Dx_%7B2%7D-x_%7B1%7Dz_%7B2%7D+%2Cx_%7B1%7Dy_%7B2%7D-y_%7B1%7Dx_%7B2%7D+%5Cright+%29)
# MAGIC 
# MAGIC **几何定义：** 我们将 ![](https://www.zhihu.com/equation?tex=%5Cvec%7Ba%7D) 和 ![](https://www.zhihu.com/equation?tex=%5Cvec%7Bb%7D) 的夹角记作 ![](https://www.zhihu.com/equation?tex=%5Ctheta) ，且 ![](https://www.zhihu.com/equation?tex=0%5Cleqslant+%5Ctheta+%5Cleqslant+%5Cpi)，那么叉乘得到的向量的模长为：
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=%7C+%5Cvec%7Ba%7D+%5Ctimes+%5Cvec%7Bb%7D+%7C+%3D+%7C+%5Cvec%7Ba%7D+%7C+%5Ccdot+%7C+%5Cvec%7Bb%7D+%7C%5Ccdot+sin%5Ctheta)
# MAGIC 
# MAGIC **方向：**与这两个向量所在平面垂直，且遵守**右手螺旋定则**（四指方向代表旋转的方向，右手四指从 ![](https://www.zhihu.com/equation?tex=%5Cvec%7Ba%7D) 转向 ![](https://www.zhihu.com/equation?tex=%5Cvec%7Bb%7D) 时，大拇指的方向即向量积的方向）
# MAGIC 
# MAGIC 用矩阵表示：
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=%5Cvec%7Ba%7D%5Ctimes+%5Cvec%7Bb%7D+%3D+%5Cbegin%7Bbmatrix%7D+0%26+-z_1%26+y_1%5C%5C+z_1%26+0%26+-x_1+%5C%5C+-y_1+%26+x_1%260+%5Cend%7Bbmatrix%7D%5Cbegin%7Bbmatrix%7D+x_2%5C%5C+y_2+%5C%5C+z_2+%5Cend%7Bbmatrix%7D%3D+%5Cleft+%28+y_%7B1%7Dz_%7B2%7D-z_%7B1%7Dy_%7B2%7D+%2Cz_%7B1%7Dx_%7B2%7D-x_%7B1%7Dz_%7B2%7D+%2Cx_%7B1%7Dy_%7B2%7D-y_%7B1%7Dx_%7B2%7D+%5Cright+%29)
# MAGIC 
# MAGIC 若为二维向量，即![](https://www.zhihu.com/equation?tex=z)的值为0，因此 ![](https://www.zhihu.com/equation?tex=%5Cvec%7Ba%7D%5Ctimes+%5Cvec%7Bb%7D+%3D%280%2C0%2C+x_%7B1%7Dy_%7B2%7D-y_%7B1%7Dx_%7B2%7D%29) ，又因为二维没有![](https://www.zhihu.com/equation?tex=z)轴，所以常写作 ![](https://www.zhihu.com/equation?tex=%5Cvec%7Ba%7D%5Ctimes+%5Cvec%7Bb%7D+%3D+x_%7B1%7Dy_%7B2%7D-y_%7B1%7Dx_%7B2%7D) ，该常量其实就是 ![](https://www.zhihu.com/equation?tex=%7C+%5Cvec%7Ba%7D+%5Ctimes+%5Cvec%7Bb%7D+%7C) 。

# COMMAND ----------

import numpy as np

a = np.array((7,8,9))
b = np.array((4,5,6))
c = np.array((1,2,3))

print(a.dot(b))
print(np.cross(a, b))


# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC **一些运算律：**
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=%5Cvec%7Ba%7D%5Ctimes+%5Cvec%7Bb%7D+%3D+-%5Cvec%7Bb%7D%5Ctimes+%5Cvec%7Ba%7D)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=%28+k%5Cvec%7Ba%7D+%29%5Ctimes+%5Cvec%7Bb%7D+%3D+k%28+%5Cvec%7Ba%7D%5Ctimes+%5Cvec%7Bb%7D+%29)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=%28+%5Cvec%7Ba%7D+%2B+%5Cvec%7Bb%7D+%29%5Ctimes+%5Cvec%7Bc%7D+%3D%5Cvec%7Ba%7D%5Ctimes+%5Cvec%7Bc%7D+%2B+%5Cvec%7Bb%7D%5Ctimes+%5Cvec%7Bc%7D)
# MAGIC 
# MAGIC **一些性质：**
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=%5Cvec%7Ba%7D%5Ctimes+%5Cvec%7Ba%7D+%3D+%5Cvec%7B0%7D) （因为![](https://www.zhihu.com/equation?tex=%5Csin0%3D0)）
# MAGIC 
# MAGIC 若两个向量互相平行，则 ![](https://www.zhihu.com/equation?tex=%5Cvec%7Ba%7D%2F%2F+%5Cvec%7Bb%7D+%5Cleftrightharpoons+%5Cvec%7Ba%7D%5Ctimes+%5Cvec%7Bb%7D+%3D+%5Cvec%7B0%7D)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=%7C+%5Cvec%7Ba%7D+%5Ctimes+%5Cvec%7Bb%7D+%7C) 的值是以 ![](https://www.zhihu.com/equation?tex=%5Cvec%7Ba%7D) 和 ![](https://www.zhihu.com/equation?tex=%5Cvec%7Bb%7D) 为边的平行四边形的面积，同样的以 ![](https://www.zhihu.com/equation?tex=%5Cvec%7Ba%7D) 和 ![](https://www.zhihu.com/equation?tex=%5Cvec%7Bb%7D) 为边的三角形的面积自然就是 ![](https://www.zhihu.com/equation?tex=%5Cfrac%7B%7C+%5Cvec%7Ba%7D+%5Ctimes+%5Cvec%7Bb%7D+%7C%7D%7B2%7D)。

# COMMAND ----------

import numpy as np

a = np.array((7,8,9))
b = np.array((4,5,6))
c = np.array((1,2,3))

print(np.cross(a, b))
print(-np.cross(b, a))


print(np.cross(5*a, b))
print(5*np.cross(a, b))

print(np.cross(a, a))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 向量的哈达玛积
# MAGIC 
# MAGIC 两个向量的哈达玛积就是对应位置的元素相乘。
# MAGIC 
# MAGIC **代数定义：** ![](https://www.zhihu.com/equation?tex=%5Cvec%7Ba%7D+*+%5Cvec%7Bb%7D+%3D+%28x_1%2Cy_1%2Cz_1%29*%28x_2%2Cy_2%2Cz_2%29%3D%28x_1x_2%2Cy_1y_2%2Cz_1z_2%29)

# COMMAND ----------

import numpy as np

a = np.array((7,8,9))
b = np.array((4,5,6))

print(a * b)

# COMMAND ----------

# MAGIC %md
# MAGIC 体会一下各种乘法的不同。

# COMMAND ----------

import numpy as np

a = np.array((7,8,9))
b = np.array((4,5,6))

print("向量和实数的积：",a * 5)
print("向量的点乘（内积）：",a.dot(b))
print("向量的叉乘（外积）：",np.cross(a,b))
print("向量的哈达玛积：",a * b)

# COMMAND ----------

# MAGIC %md
# MAGIC # 矩阵

# COMMAND ----------

# MAGIC %md
# MAGIC ## 什么是矩阵
# MAGIC 
# MAGIC **数学定义：** 矩阵是一个按照长方阵列排列的复数或实数集合。
# MAGIC 
# MAGIC 矩阵的行数和列数叫做矩阵的阶。
# MAGIC 
# MAGIC 在线性代数中：矩阵是描述线性代数中线性关系的参数，即矩阵是一个线性变换，可以将一些向量转换为另一些向量。

# COMMAND ----------

import numpy as np

matrix = np.mat(np.random.randint(1,10,(3,5)))

print(matrix)

print(type(matrix))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 矩阵的运算

# COMMAND ----------

# MAGIC %md
# MAGIC ### 矩阵的加减法
# MAGIC 
# MAGIC 矩阵的加法与减法要求进行操作的两个矩阵A和B具有相同的阶。
# MAGIC 
# MAGIC 假设矩阵A是m\*n阶，那么矩阵B也必须是m\*n阶，则C=A+B|C=A-B也是m\*n阶矩阵。
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=A%3D%5Cbegin%7Bbmatrix%7Da_%7B11%7D%26a_%7B12%7D%26%5Ccdots%26a_%7B1n%7D%5C%5Ca_%7B21%7D%26a_%7B22%7D%26%5Ccdots%26a_%7B2n%7D%5C%5C%5Cvdots%26%5Cvdots%26%5Cvdots%26%5Cvdots%5C%5Ca_%7Bm1%7D%26a_%7Bm2%7D%26%5Ccdots%26a_%7Bmn%7D%5Cend%7Bbmatrix%7D)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=B%3D%5Cbegin%7Bbmatrix%7Db_%7B11%7D%26b_%7B12%7D%26%5Ccdots%26b_%7B1n%7D%5C%5Cb_%7B21%7D%26b_%7B22%7D%26%5Ccdots%26b_%7B2n%7D%5C%5C%5Cvdots%26%5Cvdots%26%5Cvdots%26%5Cvdots%5C%5Cb_%7Bm1%7D%26b_%7Bm2%7D%26%5Ccdots%26b_%7Bmn%7D%5Cend%7Bbmatrix%7D)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=C%3DA%C2%B1B%3D%5Cbegin%7Bbmatrix%7Da_%7B11%7D%C2%B1b_%7B11%7D%26a_%7B12%7D%C2%B1b_%7B12%7D%26%5Ccdots%26a_%7B1n%7D%C2%B1b_%7B1n%7D%5C%5Ca_%7B21%7D%C2%B1b_%7B21%7D%26a_%7B22%7D%C2%B1b_%7B22%7D%26%5Ccdots%26a_%7B2n%7D%C2%B1b_%7B2n%7D%5C%5C%5Cvdots%26%5Cvdots%26%5Cvdots%26%5Cvdots%5C%5Ca_%7Bm1%7D%C2%B1b_%7Bm1%7D%26a_%7Bm2%7D%C2%B1b_%7Bm2%7D%26%5Ccdots%26a_%7Bmn%7D%C2%B1b_%7Bmn%7D%5Cend%7Bbmatrix%7D)

# COMMAND ----------

import numpy as np

A = np.mat(np.random.randint(1,10,(3,5)))
B = np.mat(np.random.randint(1,10,(3,5)))

print("A:",A)
print("B:",B)

print("A+B:",A+B)
print("A-B:",A-B)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 矩阵与数的乘法
# MAGIC 
# MAGIC 将数λ与矩阵A相乘，就是将数λ与矩阵A中的每一个元素相乘。
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=A%3D%5Cbegin%7Bbmatrix%7Da_%7B11%7D%26a_%7B12%7D%26%5Ccdots%26a_%7B1n%7D%5C%5Ca_%7B21%7D%26a_%7B22%7D%26%5Ccdots%26a_%7B2n%7D%5C%5C%5Cvdots%26%5Cvdots%26%5Cvdots%26%5Cvdots%5C%5Ca_%7Bm1%7D%26a_%7Bm2%7D%26%5Ccdots%26a_%7Bmn%7D%5Cend%7Bbmatrix%7D)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=%5Clambda+A%3D%5Cbegin%7Bbmatrix%7D%5Clambda+a_%7B11%7D%26%5Clambda+a_%7B12%7D%26%5Ccdots%26%5Clambda+a_%7B1n%7D%5C%5C%5Clambda+a_%7B21%7D%26%5Clambda+a_%7B22%7D%26%5Ccdots%26%5Clambda+a_%7B2n%7D%5C%5C%5Cvdots%26%5Cvdots%26%5Cvdots%26%5Cvdots%5C%5C%5Clambda+a_%7Bm1%7D%26%5Clambda+a_%7Bm2%7D%26%5Ccdots%26%5Clambda+a_%7Bmn%7D%5Cend%7Bbmatrix%7D)

# COMMAND ----------

import numpy as np

A = np.mat(np.random.randint(1,10,(3,5)))

print(A)

print(A * 5)

print(A * 1/5)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 矩阵与向量相乘
# MAGIC 
# MAGIC 假设![](https://www.zhihu.com/equation?tex=A)为m\*n阶矩阵，
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=A%3D%5Cbegin%7Bbmatrix%7Da_%7B11%7D%26a_%7B12%7D%26%5Ccdots%26a_%7B1n%7D%5C%5Ca_%7B21%7D%26a_%7B22%7D%26%5Ccdots%26a_%7B2n%7D%5C%5C%5Cvdots%26%5Cvdots%26%5Cvdots%26%5Cvdots%5C%5Ca_%7Bm1%7D%26a_%7Bm2%7D%26%5Ccdots%26a_%7Bmn%7D%5Cend%7Bbmatrix%7D)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=%5Cvec%7Bx%7D)是n\*1的列向量，
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=%5Cvec%7Bx%7D%3D%5Cbegin%7Bbmatrix%7Dx_1%5C%5Cx_2%5C%5C%5Cvdots%5C%5Cx_n%5Cend%7Bbmatrix%7D)
# MAGIC 
# MAGIC 则![](https://www.zhihu.com/equation?tex=A%5Cvec%7Bx%7D)是m\*1阶的列向量，记作![](https://www.zhihu.com/equation?tex=%5Cvec%7By%7D%3DA%5Cvec%7Bx%7D)。
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=%5Cvec%7By%7D%3DA%5Cvec%7Bx%7D%3D%5Cbegin%7Bbmatrix%7Dy_1%5C%5Cy_2%5C%5C%5Cvdots%5C%5Cy_n%5Cend%7Bbmatrix%7D%3D%5Cbegin%7Bbmatrix%7Da_%7B11%7Dx_1%2Ba_%7B12%7Dx_2%2B%5Ccdots%2Ba_%7B1n%7Dx_n%5C%5Ca_%7B21%7Dx_1%2Ba_%7B22%7Dx_2%2B%5Ccdots%2Ba_%7B2n%7Dx_n%5C%5C%5C%5C%5Cvdots%5C%5Ca_%7Bm1%7Dx_1%2Ba_%7Bm2%7Dx_2%2B%5Ccdots%2Ba_%7Bmn%7Dx_n%5C%5C%5Cend%7Bbmatrix%7D)

# COMMAND ----------

import numpy as np

A = np.mat([[1,2,3],[4,5,6],[7,8,9]])
x = np.array([[5],[5],[5]])

print(A)
print(x)

print("-----------")

print(A*x)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 矩阵与矩阵相乘
# MAGIC 
# MAGIC 矩阵的乘法仅当第一个矩阵A的列数与第二个矩阵B的行数相同时才能进行定义。
# MAGIC 
# MAGIC 计算方法为：矩阵C的第i行j列的元素，等于矩阵A的第i行与矩阵B的第j列的各个元素乘积之和。
# MAGIC 
# MAGIC 假设A为m\*s阶的矩阵，
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=A%3D%5Cbegin%7Bbmatrix%7Da_%7B11%7D%26a_%7B12%7D%26%5Ccdots%26a_%7B1s%7D%5C%5Ca_%7B21%7D%26a_%7B22%7D%26%5Ccdots%26a_%7B2s%7D%5C%5C%5Cvdots%26%5Cvdots%26%5Cvdots%26%5Cvdots%5C%5Ca_%7Bm1%7D%26a_%7Bm2%7D%26%5Ccdots%26a_%7Bms%7D%5Cend%7Bbmatrix%7D)
# MAGIC 
# MAGIC B为s\*n阶的矩阵，
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=B%3D%5Cbegin%7Bbmatrix%7Db_%7B11%7D%26b_%7B12%7D%26%5Ccdots%26b_%7B1n%7D%5C%5Cb_%7B21%7D%26b_%7B22%7D%26%5Ccdots%26b_%7B2n%7D%5C%5C%5Cvdots%26%5Cvdots%26%5Cvdots%26%5Cvdots%5C%5Cb_%7Bs1%7D%26b_%7Bs2%7D%26%5Ccdots%26b_%7Bsn%7D%5Cend%7Bbmatrix%7D)
# MAGIC 
# MAGIC 则C=A\*B是m\*n阶的矩阵。
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=C%3DA*B%3D%5Cbegin%7Bbmatrix%7Da_%7B11%7Db_%7B11%7D%2Ba_%7B12%7Db_%7B21%7D%2B%5Ccdots%2Ba_%7B1s%7Db_%7Bs1%7D%26a_%7B11%7Db_%7B12%7D%2Ba_%7B12%7Db_%7B22%7D%2B%5Ccdots%2Ba_%7B1s%7Db_%7Bs2%7D%26%5Ccdots%26a_%7B11%7Db_%7B1n%7D%2Ba_%7B12%7Db_%7B2n%7D%2B%5Ccdots%2Ba_%7B1s%7Db_%7Bsn%7D%5C%5Ca_%7B21%7Db_%7B11%7D%2Ba_%7B22%7Db_%7B21%7D%2B%5Ccdots%2Ba_%7B2s%7Db_%7Bs1%7D%26a_%7B21%7Db_%7B12%7D%2Ba_%7B22%7Db_%7B22%7D%2B%5Ccdots%2Ba_%7B2s%7Db_%7Bs2%7D%26%5Ccdots%26a_%7B21%7Db_%7B1n%7D%2Ba_%7B22%7Db_%7B2n%7D%2B%5Ccdots%2Ba_%7B2s%7Db_%7Bsn%7D%5C%5C%5Cvdots%26%5Cvdots%26%5Cvdots%26%5Cvdots%5C%5Ca_%7Bm1%7Db_%7B11%7D%2Ba_%7Bm2%7Db_%7B21%7D%2B%5Ccdots%2Ba_%7Bms%7Db_%7Bs1%7D%26a_%7Bm1%7Db_%7B12%7D%2Ba_%7Bm2%7Db_%7B22%7D%2B%5Ccdots%2Ba_%7Bms%7Db_%7Bs2%7D%26%5Ccdots%26a_%7Bm1%7Db_%7B1n%7D%2Ba_%7Bm2%7Db_%7B2n%7D%2B%5Ccdots%2Ba_%7Bms%7Db_%7Bsn%7D%5Cend%7Bbmatrix%7D)

# COMMAND ----------

import numpy as np

A = np.mat([[1,2,3],[4,5,6]])
B = np.mat([[1,2],[3,4],[5,6]])

print(A * B)

print(np.mat([[1,2],[3,4]]) * np.mat([[1,2],[3,4]]))

# COMMAND ----------

# MAGIC %md
# MAGIC 还有另外种相乘方式：将两个矩阵对应位置的元素相乘。这种相乘，要求矩阵A和矩阵B的行数与列数要对应相同。

# COMMAND ----------

import numpy as np

A = np.mat([[1,2],[3,4]])
B = np.mat([[1,2],[3,4]])

C = np.mat([[1,2,3],[4,5,6]])
D = np.mat([[1,2],[3,4],[5,6]])

print(A * B)
print(np.multiply(A,B))

print(C * D)
# 矩阵C与矩阵D不满足行数、列数对应相同，所以不能按对应位置元素相乘
# print(np.multiply(C,D))


# COMMAND ----------

# MAGIC %md
# MAGIC ### 矩阵的转置
# MAGIC 
# MAGIC 把矩阵A的行和列互相交换所产生的矩阵称为A的转置矩阵，这一过程称为矩阵的转置，用 ![](https://www.zhihu.com/equation?tex=A%5ET) 表示。
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=A%3D%5Cbegin%7Bbmatrix%7Da_%7B11%7D%26a_%7B12%7D%26%5Ccdots%26a_%7B1n%7D%5C%5Ca_%7B21%7D%26a_%7B22%7D%26%5Ccdots%26a_%7B2n%7D%5C%5C%5Cvdots%26%5Cvdots%26%5Cvdots%26%5Cvdots%5C%5Ca_%7Bm1%7D%26a_%7Bm2%7D%26%5Ccdots%26a_%7Bmn%7D%5Cend%7Bbmatrix%7D)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=A%5ET%3D%5Cbegin%7Bbmatrix%7Da_%7B11%7D%26a_%7B21%7D%26%5Ccdots%26a_%7Bm1%7D%5C%5Ca_%7B12%7D%26a_%7B22%7D%26%5Ccdots%26a_%7Bm2%7D%5C%5C%5Cvdots%26%5Cvdots%26%5Cvdots%26%5Cvdots%5C%5Ca_%7B1n%7D%26a_%7B2n%7D%26%5Ccdots%26a_%7Bmn%7D%5Cend%7Bbmatrix%7D)

# COMMAND ----------

import numpy as np

A = np.mat([[1,2,3],[4,5,6]])

print(A)

print("转置")
print(A.T)

# COMMAND ----------

# MAGIC %md
# MAGIC 矩阵的转置满足一些性质：
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=%28A%5ET%29^T=A)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=%28%5Clambda+A%29^T=%5Clambda+A^T)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=%28AB%29^T=B^TA^T)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=%28A%2BB%29^T=B^T%2BA^T)

# COMMAND ----------

import numpy as np

A = np.mat([[1,2,3],[4,5,6],[7,8,9]])
B = np.mat([[2,3,4],[5,6,7],[8,9,10]])

print(A)

print("矩阵的转置")
print(A.T)

print("矩阵的转置再转置，等于矩阵本身")
print((A.T).T)

print("矩阵乘数之后再转置，等于矩阵转置之后再乘以一个数")
print((5*A).T)
print(5*A.T)

print("矩阵A乘以矩阵B后转置，等于B的转置乘以A的转置")
print((A*B).T)
print(B.T*A.T)
print("注意：矩阵A乘以矩阵B后转置，不等于A的转置乘以B的转置")
print(A.T*B.T)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 矩阵的导数

# COMMAND ----------

# MAGIC %md
# MAGIC ### 导数的定义
# MAGIC 
# MAGIC 设函数![](https://www.zhihu.com/equation?tex=y%3Df%28x%29)在点![](https://www.zhihu.com/equation?tex=x_0)的邻域![](https://www.zhihu.com/equation?tex=U%28x_0%29)内有定义，当自变量![](https://www.zhihu.com/equation?tex=x)在![](https://www.zhihu.com/equation?tex=x_0)处取增量![](https://www.zhihu.com/equation?tex=%5CDelta+x)（且![](https://www.zhihu.com/equation?tex=x_0%2B%5CDelta+x%5Cin+U%28x_0%29)）时，相应地，因变量的增量![](https://www.zhihu.com/equation?tex=%5CDelta+y%3Df%28x_0%2B%5CDelta+x%29%2Df%28x_0%29)，若![](https://www.zhihu.com/equation?tex=%5CDelta+y)与![](https://www.zhihu.com/equation?tex=%5CDelta+x)之比当![](https://www.zhihu.com/equation?tex=%5CDelta+x%E2%86%920)时的极限存在，那么称函数![](https://www.zhihu.com/equation?tex=y%3Df%28x%29)在点![](https://www.zhihu.com/equation?tex=x_0)处可导，并称这个极限为函数![](https://www.zhihu.com/equation?tex=y%3Df%28x%29)在点![](https://www.zhihu.com/equation?tex=x_0)处的导数，记为![](https://www.zhihu.com/equation?tex=f%27%28x_0%29)。即：
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=f%27%28x_0%29%3D%5Clim_%7B%5CDelta+x%5Crightarrow+0%7D%5Cfrac%7B%5CDelta+y%7D%7B%5CDelta+x%7D%3D%5Cfrac%7Bf%28x_0%2B%5CDelta+x%29-f%28x_0%29%7D%7B%5CDelta+x%7D)
# MAGIC 
# MAGIC 或：
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=f%27%28x_0%29%3D%5Clim_%7Bx%5Crightarrow+x_0%7D%5Cfrac%7Bf%28x%29-f%28x_0%29%7D%7Bx-x_0%7D)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 常见的一些导数
# MAGIC 
# MAGIC | ![](https://www.zhihu.com/equation?tex=f%28x%29) |  ![](https://www.zhihu.com/equation?tex=f%27%28x%29) |
# MAGIC | ---------- | ---------- |
# MAGIC | C | 0 |
# MAGIC | ![](https://www.zhihu.com/equation?tex=x%5En) | ![](https://www.zhihu.com/equation?tex=nx%5E%7Bn-1%7D) |
# MAGIC | ![](https://www.zhihu.com/equation?tex=%5Cfrac%7B1%7D%7Bx%7D) | ![](https://www.zhihu.com/equation?tex=-%5Cfrac%7B1%7D%7Bx%5E2%7D) |
# MAGIC | ![](https://www.zhihu.com/equation?tex=e%5Ex) | ![](https://www.zhihu.com/equation?tex=e%5Ex) |
# MAGIC | ![](https://www.zhihu.com/equation?tex=a%5Ex) | ![](https://www.zhihu.com/equation?tex=a%5Ex%5Cln+a) |
# MAGIC | ![](https://www.zhihu.com/equation?tex=%5Cln+x) | ![](https://www.zhihu.com/equation?tex=%5Cfrac%7B1%7D%7Bx%7D) |
# MAGIC | ![](https://www.zhihu.com/equation?tex=%5Csin+x) | ![](https://www.zhihu.com/equation?tex=%5Ccos+x) |
# MAGIC | ![](https://www.zhihu.com/equation?tex=%5Ccos+x) | ![](https://www.zhihu.com/equation?tex=-%5Csin+x) |

# COMMAND ----------

# MAGIC %md
# MAGIC ### 矩阵的导数
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=%5Cfrac%7B%5CDelta+AX%7D%7B%5CDelta+X%7D%3DA%5ET)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=%5Cfrac%7B%5CDelta+X%5ETAX%7D%7B%5CDelta+X%7D%3D%28A%2BA%5ET%29X)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 几种特殊的矩阵

# COMMAND ----------

# MAGIC %md
# MAGIC ### 零矩阵
# MAGIC 矩阵中各个元素全为0的矩阵。

# COMMAND ----------

import numpy as np

A = np.mat(np.zeros((3,4)))
B = np.mat(np.ones((3,4)))

print(A)
print(B)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 单位矩阵
# MAGIC 主对角线的元素都为1，其余元素都为0的矩阵。

# COMMAND ----------

import numpy as np

A = np.mat(np.identity(5))

print(A)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 对角矩阵
# MAGIC 主对角线以外的元素都为0，主对角线上的元素不为0。

# COMMAND ----------

import numpy as np

A = np.mat(np.diag((1,2,3,4)))

print(A)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 对称矩阵
# MAGIC 以主对角线为对称轴，各个元素对应相等的矩阵。
# MAGIC 
# MAGIC 对称矩阵的一个性质：![](https://www.zhihu.com/equation?tex=X%5ET%3DX)

# COMMAND ----------

import numpy as np

X = np.mat([[1,2,3],[4,5,6],[7,8,9]])
A = np.triu(X)
B = np.diagonal(X)
C = np.diag(B)

D = A + A.T - C

print("元素矩阵")
print(X)
print("上三角矩阵")
print(A)
print("主对角线元素")
print(B)
print("对角矩阵")
print(C)
print("对称矩阵")
print(D)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 正交矩阵
# MAGIC 如果矩阵满足：![](https://www.zhihu.com/equation?tex=AA%5ET=A%5ETA=E)(单位矩阵)，则称该矩阵为正交矩阵。

# COMMAND ----------

# MAGIC %md
# MAGIC ### 正定矩阵
# MAGIC ![](https://www.zhihu.com/equation?tex=f%28x%29%3Dx_1%5E2%2Bx_2%5E2%2b0.1x_1x_2)
# MAGIC 
# MAGIC 矩阵A：![](https://www.zhihu.com/equation?tex=%5Cbegin%7Bbmatrix%7D++1%260.05+%5C%5C++0.05%261++%5Cend%7Bbmatrix%7D)
# MAGIC 
# MAGIC 效果：具有唯一的最小值。

# COMMAND ----------

# MAGIC %md
# MAGIC ### 半正定矩阵
# MAGIC ![](https://www.zhihu.com/equation?tex=f%28x%29%3Dx_1%5E2%2Bx_2%5E2%2b2x_1x_2)
# MAGIC 
# MAGIC 矩阵A：![](https://www.zhihu.com/equation?tex=%5Cbegin%7Bbmatrix%7D++1%261+%5C%5C++1%261++%5Cend%7Bbmatrix%7D)
# MAGIC 
# MAGIC 效果：具有不唯一的最小值。

# COMMAND ----------

# MAGIC %md
# MAGIC # 最小二乘法

# COMMAND ----------

# MAGIC %md
# MAGIC ## 核心
# MAGIC 求解未知参数，使得测量值与真实值之差（误差、残差）的平方和达到最小（损失函数达到最小）。

# COMMAND ----------

# MAGIC %md
# MAGIC ## 目标函数
# MAGIC ![](https://www.zhihu.com/equation?tex=E%3D%5Csum_%7Bi%3D1%7D%5En%28y-y_i%29%5E2)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 目的
# MAGIC 求最小值

# COMMAND ----------

# MAGIC %md
# MAGIC ## 解法
# MAGIC 
# MAGIC 1.列出目标函数E（损失函数![](https://www.zhihu.com/equation?tex=j%28%CE%B8%29)），此时的预测值用![](https://www.zhihu.com/equation?tex=y%3Dh_%CE%B8%28x%5E%7B%28i%29%7D%29%3Dx%CE%B8)来表示。
# MAGIC 
# MAGIC 2.求损失函数关于参数的导数，使导数为0，代表损失函数最小。
# MAGIC 
# MAGIC 3.此时的参数即为我们所求未知解的参数。

# COMMAND ----------

# MAGIC %md
# MAGIC ## 求解过程
# MAGIC 首先明确一下向量和矩阵的导数：
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=%5Cfrac%7B%5CDelta+AX%7D%7B%5CDelta+X%7D%3DA%5ET)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=%5Cfrac%7B%5CDelta+X%5ETAX%7D%7B%5CDelta+X%7D%3D%28A%2BA%5ET%29X)
# MAGIC 
# MAGIC 一个条件：要满足正定矩阵
