# Databricks notebook source
# MAGIC %md
# MAGIC # 向量

# COMMAND ----------

# MAGIC %md
# MAGIC ## 什么是向量
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=%5Cvec%7Ba%7D)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=%5Cvec%7BAB%7D)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=%5Cvec%7Ba%7D%3D%5Cbegin%7Bbmatrix%7Dx%5C%5Cy%5Cend%7Bbmatrix%7D)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=%5Cvec%7Ba%7D%3D%5Cbegin%7Bbmatrix%7Dx%5C%5Cy%5C%5Cz%5Cend%7Bbmatrix%7D)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=%5Cleft+%7C+%5Cvec%7Ba%7D+%5Cright+%7C)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=%5Cvec%7Ba%7D+%3D+%5Cleft+%28+x%2Cy%2Cz+%5Cright+%29)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=%5Cleft+%7C+%5Cvec%7Ba%7D+%5Cright+%7C+%3D+%5Csqrt%7Bx%5E2+%2B+y%5E2%2B+z%5E2%7D)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=%5Chat%7Ba%7D)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=%5Chat%7Ba%7D+%3D+%5Cfrac%7B%5Cvec%7Ba%7D%7D%7B%7C+%5Cvec%7Ba%7D%7C%7D)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=-%5Cvec%7Ba%7D)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=%5Cvec%7Ba%7D+%2F%2F+%5Cvec%7Bb%7D)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 向量的运算
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=%5Cvec%7Ba%7D%3D%28x_1%2Cy_1%2Cz_1%29)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=%5Cvec%7Bb%7D%3D%28x_2%2Cy_2%2Cz_2%29)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 向量的加法
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=%5Cvec%7BOA%7D+%2B+%5Cvec%7BOB%7D+%3D+%5Cvec%7BOC%7D)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=%5Cvec%7Ba%7D+%2B+%5Cvec%7Bb%7D+%3D+%28x_1%2Cy_1%2Cz_1%29%2B%28x_2%2Cy_2%2Cz_2%29%3D%28x_1%2Bx_2%2Cy_1%2By_2%2Cz_1%2Bz_2%29)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=%5Cvec%7Ba%7D+%2B+%5Cvec%7Bb%7D+%3D+%5Cbegin%7Bbmatrix%7D+x_%7B1%7D%5C%5C+y_%7B1%7D+%5C%5Cz_1%5Cend%7Bbmatrix%7D+%2B+%5Cbegin%7Bbmatrix%7D+x_%7B2%7D%5C%5C+y_%7B2%7D+%5C%5Cz_2%5Cend%7Bbmatrix%7D+%3D+%5Cbegin%7Bbmatrix%7D+x_%7B1%7D+%2B+x_%7B2%7D%5C%5C+y_%7B1%7D+%2B+y_%7B2%7D+%5C%5Cz_1%2Bz_2%5Cend%7Bbmatrix%7D)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=%5Cvec%7Ba%7D+%2B+0+%3D+0+%2B+%5Cvec%7Ba%7D+%3D+%5Cvec%7Ba%7D)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=%5Cvec%7Ba%7D+%2B+%5Cvec%7Bb%7D+%3D+%5Cvec%7Bb%7D+%2B+%5Cvec%7Ba%7D)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=%5Cleft+%28+%5Cvec%7Ba%7D+%2B+%5Cvec%7Bb%7D+%5Cright%29+%2B+%5Cvec%7Bc%7D+%3D+%5Cvec%7Ba%7D+%2B+%5Cleft+%28+%5Cvec%7Bb%7D+%2B+%5Cvec%7Bc%7D+%5Cright%29)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 向量的减法
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=%5Cvec%7BOA%7D+-+%5Cvec%7BOB%7D+%3D+%5Cvec%7BBA%7D)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=%5Cvec%7Ba%7D+-+%5Cvec%7Bb%7D+%3D+%28x_1%2Cy_1%2Cz_1%29-%28x_2%2Cy_2%2Cz_2%29%3D%28x_1-x_2%2Cy_1-y_2%2Cz_1-z_2%29)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=%5Cvec%7Ba%7D+-+%5Cvec%7Bb%7D+%3D+%5Cbegin%7Bbmatrix%7D+x_%7B1%7D%5C%5C+y_%7B1%7D+%5C%5Cz_1%5Cend%7Bbmatrix%7D+-+%5Cbegin%7Bbmatrix%7D+x_%7B2%7D%5C%5C+y_%7B2%7D%5C%5Cz_2+%5Cend%7Bbmatrix%7D+%3D+%5Cbegin%7Bbmatrix%7D+x_%7B1%7D+-+x_%7B2%7D%5C%5C+y_%7B1%7D+-+y_%7B2%7D+%5C%5Cz_1-z_2%5Cend%7Bbmatrix%7D)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=%5Cvec%7Ba%7D%2B%28-%5Cvec%7Bb%7D%29%3D%5Cvec%7Ba%7D-%5Cvec%7Bb%7D)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 实数和向量的积
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=k%5Cvec%7Ba%7D)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=+%7C+k+%5Cvec%7Ba%7D+%7C+%3D++%7C+k%7C+%2A+%7C+%5Cvec%7Ba%7D+%7C)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=k%5Cvec%7Ba%7D%3D0)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=%5Cvec%7Ba%7D%3D0)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=%28+k%5Cvec%7Ba%7D+%29%5Ccdot+%5Cvec%7Bb%7D+%3D+k%28%5Cvec%7Ba%7D+%5Ccdot+%5Cvec%7Bb%7D+%29+%3D+%5Cvec%7Ba%7D+%5Ccdot+%28+k%5Cvec%7Bb%7D+%29)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=%28+j%2B+k%29+%5Cvec%7Ba%7D+%3D+j%5Cvec%7Ba%7D+%2B+k%5Cvec%7Ba%7D)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=k%28+%5Cvec%7Ba%7D+%2B+%5Cvec%7Bb%7D+%29+%3Dk%5Cvec%7Ba%7D+%2Bk%5Cvec%7Bb%7D)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 向量的点乘（点积，内积，数量积）
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=%5Cvec%7Ba%7D%5Ccdot+%5Cvec%7Bb%7D+%3D+x_%7B1%7Dx_%7B2%7D%2By_%7B1%7Dy_%7B2%7D%2Bz_1z_2)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=%5Ctheta)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=0%5Cleqslant+%5Ctheta+%5Cleqslant+%5Cpi)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=%5Cvec%7Ba%7D%5Ccdot+%5Cvec%7Bb%7D+%3D+%7C+%5Cvec%7Ba%7D+%7C+%5Ccdot+%7C+%5Cvec%7Bb%7D+%7C%5Ccdot+cos%5Ctheta)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=%5Cvec%7Ba%7D%5Ccdot+%5Cvec%7Bb%7D+%3D+%5Cpm+%5Cleft+%7C+%5Cvec%7Ba%7D+%5Cright+%7C+%5Ccdot+%5Cleft+%7C+%5Cvec%7Bb%7D+%5Cright+%7C)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=%5Ctheta%3D0)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=%5Ccos%5Ctheta%3D1)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=%5Ctheta%3D%5Cpi)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=%5Ccos%5Ctheta%3D-1)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=%5Cvec%7Ba%7D%5Ccdot+%5Cvec%7Bb%7D%3D%5Cvec%7Bb%7D%5Ccdot+%5Cvec%7Ba%7D)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=%28+k%5Cvec%7Ba%7D+%29%5Ccdot+%5Cvec%7Bb%7D+%3D+k%28+%5Cvec%7Ba%7D%5Ccdot+%5Cvec%7Bb%7D+%29)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=%28+%5Cvec%7Ba%7D+%2B+%5Cvec%7Bb%7D+%29%5Ccdot+%5Cvec%7Bc%7D+%3D%5Cvec%7Ba%7D%5Ccdot+%5Cvec%7Bc%7D+%2B+%5Cvec%7Bb%7D%5Ccdot+%5Cvec%7Bc%7D)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=%5Cvec%7Ba%7D%5Ccdot+%5Cvec%7Ba%7D+%3D+%7C+%5Cvec%7Ba%7D+%7C+%5E%7B2%7D)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=%5Ccos%5Ctheta%3D0)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=%5Cvec%7Ba%7D%5Cperp+%5Cvec%7Bb%7D+%5Cleftrightharpoons+%5Cvec%7Ba%7D%5Ccdot+%5Cvec%7Bb%7D+%3D+0)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 向量的叉乘（叉积，外积，向量积）
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=%5Cvec%7Ba%7D%5Ctimes+%5Cvec%7Bb%7D)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=%5Cvec%7Ba%7D+%5Cwedge+%5Cvec%7Bb%7D)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=%5Cvec%7Ba%7D%5Ctimes+%5Cvec%7Bb%7D%3D+%5Cleft+%28+y_%7B1%7Dz_%7B2%7D-z_%7B1%7Dy_%7B2%7D+%2Cz_%7B1%7Dx_%7B2%7D-x_%7B1%7Dz_%7B2%7D+%2Cx_%7B1%7Dy_%7B2%7D-y_%7B1%7Dx_%7B2%7D+%5Cright+%29)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=%7C+%5Cvec%7Ba%7D+%5Ctimes+%5Cvec%7Bb%7D+%7C+%3D+%7C+%5Cvec%7Ba%7D+%7C+%5Ccdot+%7C+%5Cvec%7Bb%7D+%7C%5Ccdot+sin%5Ctheta)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=%5Cvec%7Ba%7D%5Ctimes+%5Cvec%7Bb%7D+%3D+%5Cbegin%7Bbmatrix%7D+0%26+-z_1%26+y_1%5C%5C+z_1%26+0%26+-x_1+%5C%5C+-y_1+%26+x_1%260+%5Cend%7Bbmatrix%7D%5Cbegin%7Bbmatrix%7D+x_2%5C%5C+y_2+%5C%5C+z_2+%5Cend%7Bbmatrix%7D%3D+%5Cleft+%28+y_%7B1%7Dz_%7B2%7D-z_%7B1%7Dy_%7B2%7D+%2Cz_%7B1%7Dx_%7B2%7D-x_%7B1%7Dz_%7B2%7D+%2Cx_%7B1%7Dy_%7B2%7D-y_%7B1%7Dx_%7B2%7D+%5Cright+%29)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=%5Cvec%7Ba%7D%5Ctimes+%5Cvec%7Bb%7D+%3D%280%2C0%2C+x_%7B1%7Dy_%7B2%7D-y_%7B1%7Dx_%7B2%7D%29)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=%5Cvec%7Ba%7D%5Ctimes+%5Cvec%7Bb%7D+%3D+x_%7B1%7Dy_%7B2%7D-y_%7B1%7Dx_%7B2%7D)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=%5Cvec%7Ba%7D%5Ctimes+%5Cvec%7Bb%7D+%3D+-%5Cvec%7Bb%7D%5Ctimes+%5Cvec%7Ba%7D)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=%28+k%5Cvec%7Ba%7D+%29%5Ctimes+%5Cvec%7Bb%7D+%3D+k%28+%5Cvec%7Ba%7D%5Ctimes+%5Cvec%7Bb%7D+%29)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=%28+%5Cvec%7Ba%7D+%2B+%5Cvec%7Bb%7D+%29%5Ctimes+%5Cvec%7Bc%7D+%3D%5Cvec%7Ba%7D%5Ctimes+%5Cvec%7Bc%7D+%2B+%5Cvec%7Bb%7D%5Ctimes+%5Cvec%7Bc%7D)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=%5Cvec%7Ba%7D%5Ctimes+%5Cvec%7Ba%7D+%3D+%5Cvec%7B0%7D)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=%5Csin0%3D0)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=%5Cvec%7Ba%7D%2F%2F+%5Cvec%7Bb%7D+%5Cleftrightharpoons+%5Cvec%7Ba%7D%5Ctimes+%5Cvec%7Bb%7D+%3D+%5Cvec%7B0%7D)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=%7C+%5Cvec%7Ba%7D+%5Ctimes+%5Cvec%7Bb%7D+%7C)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=%5Cfrac%7B%7C+%5Cvec%7Ba%7D+%5Ctimes+%5Cvec%7Bb%7D+%7C%7D%7B2%7D)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 向量的哈达玛积
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=%5Cvec%7Ba%7D+*+%5Cvec%7Bb%7D+%3D+%28x_1%2Cy_1%2Cz_1%29*%28x_2%2Cy_2%2Cz_2%29%3D%28x_1x_2%2Cy_1y_2%2Cz_1z_2%29)

# COMMAND ----------

# MAGIC %md
# MAGIC # 矩阵
# MAGIC 
# MAGIC 矩阵

# COMMAND ----------

# MAGIC %md
# MAGIC ## 矩阵的运算
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=A%3D%5Cbegin%7Bbmatrix%7Da_%7B11%7D%26a_%7B12%7D%26%5Ccdots%26a_%7B1n%7D%5C%5Ca_%7B21%7D%26a_%7B22%7D%26%5Ccdots%26a_%7B2n%7D%5C%5C%5Cvdots%26%5Cvdots%26%5Cvdots%26%5Cvdots%5C%5Ca_%7Bm1%7D%26a_%7Bm2%7D%26%5Ccdots%26a_%7Bmn%7D%5Cend%7Bbmatrix%7D)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=B%3D%5Cbegin%7Bbmatrix%7Db_%7B11%7D%26b_%7B12%7D%26%5Ccdots%26b_%7B1n%7D%5C%5Cb_%7B21%7D%26b_%7B22%7D%26%5Ccdots%26b_%7B2n%7D%5C%5C%5Cvdots%26%5Cvdots%26%5Cvdots%26%5Cvdots%5C%5Cb_%7Bm1%7D%26b_%7Bm2%7D%26%5Ccdots%26b_%7Bmn%7D%5Cend%7Bbmatrix%7D)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 矩阵的加减法
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=C%3DA%C2%B1B%3D%5Cbegin%7Bbmatrix%7Da_%7B11%7D%C2%B1b_%7B11%7D%26a_%7B12%7D%C2%B1b_%7B12%7D%26%5Ccdots%26a_%7B1n%7D%C2%B1b_%7B1n%7D%5C%5Ca_%7B21%7D%C2%B1b_%7B21%7D%26a_%7B22%7D%C2%B1b_%7B22%7D%26%5Ccdots%26a_%7B2n%7D%C2%B1b_%7B2n%7D%5C%5C%5Cvdots%26%5Cvdots%26%5Cvdots%26%5Cvdots%5C%5Ca_%7Bm1%7D%C2%B1b_%7Bm1%7D%26a_%7Bm2%7D%C2%B1b_%7Bm2%7D%26%5Ccdots%26a_%7Bmn%7D%C2%B1b_%7Bmn%7D%5Cend%7Bbmatrix%7D)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 矩阵与数的乘法
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=%5Clambda+A%3D%5Cbegin%7Bbmatrix%7D%5Clambda+a_%7B11%7D%26%5Clambda+a_%7B12%7D%26%5Ccdots%26%5Clambda+a_%7B1n%7D%5C%5C%5Clambda+a_%7B21%7D%26%5Clambda+a_%7B22%7D%26%5Ccdots%26%5Clambda+a_%7B2n%7D%5C%5C%5Cvdots%26%5Cvdots%26%5Cvdots%26%5Cvdots%5C%5C%5Clambda+a_%7Bm1%7D%26%5Clambda+a_%7Bm2%7D%26%5Ccdots%26%5Clambda+a_%7Bmn%7D%5Cend%7Bbmatrix%7D)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 矩阵与向量相乘
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=%5Cvec%7Bx%7D%3D%5Cbegin%7Bbmatrix%7Dx_1%5C%5Cx_2%5C%5C%5Cvdots%5C%5Cx_n%5Cend%7Bbmatrix%7D)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=%5Cvec%7By%7D%3DA%5Cvec%7Bx%7D%3D%5Cbegin%7Bbmatrix%7Dy_1%5C%5Cy_2%5C%5C%5Cvdots%5C%5Cy_n%5Cend%7Bbmatrix%7D%3D%5Cbegin%7Bbmatrix%7Da_%7B11%7Dx_1%2Ba_%7B12%7Dx_2%2B%5Ccdots%2Ba_%7B1n%7Dx_n%5C%5Ca_%7B21%7Dx_1%2Ba_%7B22%7Dx_2%2B%5Ccdots%2Ba_%7B2n%7Dx_n%5C%5C%5C%5C%5Cvdots%5C%5Ca_%7Bm1%7Dx_1%2Ba_%7Bm2%7Dx_2%2B%5Ccdots%2Ba_%7Bmn%7Dx_n%5C%5C%5Cend%7Bbmatrix%7D)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 矩阵与矩阵相乘
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=A%3D%5Cbegin%7Bbmatrix%7Da_%7B11%7D%26a_%7B12%7D%26%5Ccdots%26a_%7B1s%7D%5C%5Ca_%7B21%7D%26a_%7B22%7D%26%5Ccdots%26a_%7B2s%7D%5C%5C%5Cvdots%26%5Cvdots%26%5Cvdots%26%5Cvdots%5C%5Ca_%7Bm1%7D%26a_%7Bm2%7D%26%5Ccdots%26a_%7Bms%7D%5Cend%7Bbmatrix%7D)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=B%3D%5Cbegin%7Bbmatrix%7Db_%7B11%7D%26b_%7B12%7D%26%5Ccdots%26b_%7B1n%7D%5C%5Cb_%7B21%7D%26b_%7B22%7D%26%5Ccdots%26b_%7B2n%7D%5C%5C%5Cvdots%26%5Cvdots%26%5Cvdots%26%5Cvdots%5C%5Cb_%7Bs1%7D%26b_%7Bs2%7D%26%5Ccdots%26b_%7Bsn%7D%5Cend%7Bbmatrix%7D)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=C%3DA*B%3D%5Cbegin%7Bbmatrix%7Da_%7B11%7Db_%7B11%7D%2Ba_%7B12%7Db_%7B21%7D%2B%5Ccdots%2Ba_%7B1s%7Db_%7Bs1%7D%26a_%7B11%7Db_%7B12%7D%2Ba_%7B12%7Db_%7B22%7D%2B%5Ccdots%2Ba_%7B1s%7Db_%7Bs2%7D%26%5Ccdots%26a_%7B11%7Db_%7B1n%7D%2Ba_%7B12%7Db_%7B2n%7D%2B%5Ccdots%2Ba_%7B1s%7Db_%7Bsn%7D%5C%5Ca_%7B21%7Db_%7B11%7D%2Ba_%7B22%7Db_%7B21%7D%2B%5Ccdots%2Ba_%7B2s%7Db_%7Bs1%7D%26a_%7B21%7Db_%7B12%7D%2Ba_%7B22%7Db_%7B22%7D%2B%5Ccdots%2Ba_%7B2s%7Db_%7Bs2%7D%26%5Ccdots%26a_%7B21%7Db_%7B1n%7D%2Ba_%7B22%7Db_%7B2n%7D%2B%5Ccdots%2Ba_%7B2s%7Db_%7Bsn%7D%5C%5C%5Cvdots%26%5Cvdots%26%5Cvdots%26%5Cvdots%5C%5Ca_%7Bm1%7Db_%7B11%7D%2Ba_%7Bm2%7Db_%7B21%7D%2B%5Ccdots%2Ba_%7Bms%7Db_%7Bs1%7D%26a_%7Bm1%7Db_%7B12%7D%2Ba_%7Bm2%7Db_%7B22%7D%2B%5Ccdots%2Ba_%7Bms%7Db_%7Bs2%7D%26%5Ccdots%26a_%7Bm1%7Db_%7B1n%7D%2Ba_%7Bm2%7Db_%7B2n%7D%2B%5Ccdots%2Ba_%7Bms%7Db_%7Bsn%7D%5Cend%7Bbmatrix%7D)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 矩阵的转置
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=A^T%3D%5Cbegin%7Bbmatrix%7Da_%7B11%7D%26a_%7B21%7D%26%5Ccdots%26a_%7Bm1%7D%5C%5Ca_%7B12%7D%26a_%7B22%7D%26%5Ccdots%26a_%7Bm2%7D%5C%5C%5Cvdots%26%5Cvdots%26%5Cvdots%26%5Cvdots%5C%5Ca_%7B1n%7D%26a_%7B2n%7D%26%5Ccdots%26a_%7Bmn%7D%5Cend%7Bbmatrix%7D)
