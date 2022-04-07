# Databricks notebook source
# MAGIC %md
# MAGIC # 概率

# COMMAND ----------

# MAGIC %md
# MAGIC ## 随机变量

# COMMAND ----------

# MAGIC %md
# MAGIC ### 离散型随机变量

# COMMAND ----------

# MAGIC %md
# MAGIC #### 概率质量函数 (Probability Mass Function，PMF)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=P)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=X)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=X%3Dx)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=P%28x%29)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=P%28X%3Dx%29)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=X%5Csim+P%28x%29)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=P%28x_i%29%5Cgeq+0)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=%5Csum_%7Bi%3D1%7D%5E%5Cinfty+P%28x_i%29%3D1)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=%5Cbegin%7Barray%7D%7Bc%7Cc%7D+%5Ctext%7BX%7D+%26+0%26+1+%5C%5C+%5Chline+P+%261-p%26p+%5Cend%7Barray%7D)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 概率质量函数的特性

# COMMAND ----------

# MAGIC %md
# MAGIC #### 联合概率分布

# COMMAND ----------

# MAGIC %md
# MAGIC #### 伯努利试验

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 典例

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 相关定理
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=P_n%28k%29%3DC_n^kp^k%281-p%29^%7Bn-k%7D%5Cquad%28k%3D0,1,2,%5Ccdots,n%29)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 推论
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=p%281-p%29^%7Bk-1%7D%5Cquad%28k%3D1,2,%5Ccdots%29)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 常用离散分布

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 二项分布
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=A)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=X%3D0%2C1%2C2%2C%5Ccdots%2Cn.)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=p)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=P%28X%3Dk%29%3D%5Cleft%28%5Cbegin%7Barray%7D%7Bl%7D%7Bn%7D+%5C%5C+%7Bk%7D%5Cend%7Barray%7D%5Cright%29+p%5E%7Bk%7D%281-p%29%5E%7Bn-k%7D%2C+%5Cquad+k%3D0%2C1%2C+%5Ccdots%2C+n.)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=X%5Csim+b%28n%2Cp%29)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=n%2Cp)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=np)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=np%281-p%29)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 两点分布
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=n%3D1)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=X%3D0%2C1)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=P%28X%3Dx%29%3Dp%5E%7Bx%7D%281-p%29%5E%7B1-x%7D%2C%5Cquad+x%3D0%2C1)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=%5Cbegin%7Barray%7D%7Bc%7Cc%7D+%5Ctext%7BX%7D+%26+0%26+1+%5C%5C+%5Chline+P+%261-p%26p+%5Cend%7Barray%7D)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=p%3DP%28X%3D1%29)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=p)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=p%281-p%29)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 负二项分布
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=r)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=r%2Cr%2B1%2C%5Ccdots%2Cr%2Bm%2C%5Ccdots)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=P%28X%3Dk%29%3D%5Cleft%28%5Cbegin%7Barray%7D%7Bl%7D%7Bk-1%7D+%5C%5C+%7Br-1%7D%5Cend%7Barray%7D%5Cright%29+p%5Er%5C%2C%281-p%29%5E%7Bk-r%7D%2C+%5Cquad+k%3Dr%2C+r%2B1%2C+%5Ccdots)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=X%5Csim+Nb%28r%2Cp%29)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=r%3D1)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=%5Cfrac%7Br%7D%7Bp%7D)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=%5Cfrac%7Br%281-p%29%7D%7Bp%5E2%7D)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 几何分布
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=X%3D1%2C2%2C%5Ccdots)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=P%28X%3Dk%29%3D%281-p%29%5E%7Bk-1%7D+p%2C+%5Cquad+k%3D1%2C2%2C+%5Ccdots)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=X%5Csim+Ge%28p%29)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=%5Cfrac%7B1%7D%7Bp%7D)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=%5Cfrac%7B1-p%7D%7Bp%5E2%7D)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 超几何分布
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=P%28X%3Dk%29%3D%5Cfrac%7B%5Cbinom%7BM%7D%7Bk%7D%5Cbinom%7BN-M%7D%7Bn-k%7D+%7D%7B%5Cbinom%7BN%7D%7Bn%7D%7D%2C%5Cquad+k%3D0%2C1%2C+%5Ccdots%2C+r)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=X%5Csim+h%28n%2CN%2CM%29)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=r%3D%5Cmin+%5C%7BM%2C+n%5C%7D)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=M+%5Cleqslant+N%2C+n+%5Cleqslant+N)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=n%2C+N%2C+M)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=n%5Cfrac%7BM%7D%7BN%7D)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=%5Cfrac%7BnM%28N-M%29%28N-n%29%7D%7BN%5E2%28N-1%29%7D)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 泊松分布
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=P%28X%3Dk%29%3D%5Cfrac%7B%5Clambda%5E%7Bk%7D%7D%7Bk+%21%7D+%5Cmathrm%7Be%7D%5E%7B-%5Clambda%7D%2C%5Cquad+k%3D0%2C1%2C2%2C+%5Ccdots)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=X%5Csim+P%28%5Clambda%29)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=%5Clambda)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 连续型随机变量

# COMMAND ----------

# MAGIC %md
# MAGIC #### 概率密度函数
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=f%28x%29)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=P%28x%29%3D%5Cint_%7B-%5Cinfty%7D%5E%7Bx%7Df%28t%29%7B%5Crm+d%7Dt)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 概率密度函数的特性
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=f%28x%29%5Cgeq0)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=%5Cint_%7B-%5Cinfty%7D%5E%7B%2B%5Cinfty%7Df%28x%29%7B%5Crm+d%7Dx%3D1)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=P%28a%5Clt+x%5Cleq+b%29%3D%5Cint_%7Ba%7D%5E%7Bb%7Df%28x%29%7B%5Crm+d%7Dx)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=F%28x%29%3D%5Cint_%7B-%5Cinfty%7D%5E%7Bx%7Dp%28t%29%7B%5Crm+d%7Dt%3D%5Cint_%7B-%5Cinfty%7D%5E%7Bx%7D%5Cfrac%7B1%7D%7B%5Csqrt%7B2+%5Cpi%7D+%5Csigma%7D+%5Cmathrm%7Be%7D%5E%7B-%5Cfrac%7B%28t-%5Cmu%29%5E%7B2%7D%7D%7B2+%5Csigma%5E%7B2%7D%7D%7D%7B%5Crm+d%7Dt)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 常用连续分布

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 正态分布
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=p%28x%29%3D%5Cfrac%7B1%7D%7B%5Csqrt%7B2+%5Cpi%7D+%5Csigma%7D+%5Cmathrm%7Be%7D%5E%7B-%5Cfrac%7B%28x-%5Cmu%29%5E%7B2%7D%7D%7B2+%5Csigma%5E%7B2%7D%7D%7D%2C%5Cquad+-%5Cinfty%3Cx%3C%5Cinfty)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=%5Cmu)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=%5Csigma)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=%5Csigma^2)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=X%5Csim+N%28%5Cmu%2C%5Csigma%5E2%29)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=F%28x%29%3D%5Cint_%7B-%5Cinfty%7D%5E%7Bx%7Dp%28t%29%7B%5Crm+d%7Dt%3D%5Cint_%7B-%5Cinfty%7D%5E%7Bx%7D%5Cfrac%7B1%7D%7B%5Csqrt%7B2+%5Cpi%7D+%5Csigma%7D+%5Cmathrm%7Be%7D%5E%7B-%5Cfrac%7B%28t-%5Cmu%29%5E%7B2%7D%7D%7B2+%5Csigma%5E%7B2%7D%7D%7D%7B%5Crm+d%7Dt)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=P%28X%5Cleq+c%29%3D%5CPhi%5Cleft%28%5Cfrac%7Ba-%5Cmu%7D%7B%5Csigma%7D%5Cright%29)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=P%28a%3Cx%5Cleq+b%29%3D%5CPhi%5Cleft%28%5Cfrac%7Bb-%5Cmu%7D%7B%5Csigma%7D%5Cright%29-%5CPhi%5Cleft%28%5Cfrac%7Ba-%5Cmu%7D%7B%5Csigma%7D%5Cright%29)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 标准正态分布
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=%5Cmu%3D0%2C%5Csigma%5E2%3D1)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=X%5Csim+N%280%2C1%29)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=%5Cvarphi%28x%29%3D%5Cfrac%7B1%7D%7B%5Csqrt%7B2%5Cpi%7D%7D%5C%2C%7B%5Crm+e%7D%5E%7B-%5Cfrac%7Bx%5E2%7D%7B2%7D%7D)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=%5CPhi%28x%29%3D%5Cint_%7B-%5Cinfty%7D%5E%7Bx%7D%5Cvarphi%28t%29%7B%5Crm+d%7Dt%3D%5Cint_%7B-%5Cinfty%7D%5E%7Bx%7D%5Cfrac%7B1%7D%7B%5Csqrt%7B2%5Cpi%7D%7D%5C%2C%7B%5Crm+e%7D%5E%7B-%5Cfrac%7Bt%5E2%7D%7B2%7D%7D%7B%5Crm+d%7Dt)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=X%5E%2A%3D%5Cfrac%7BX-%5Cmu%7D%7B%5Csigma%7D%5Csim+N%280%2C1%29)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=X%5E%2A)
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
# MAGIC ##### 均匀分布
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=p%28x%29%3D%5Cbegin%7Bcases%7D+%5Cfrac%7B1%7D%7Bb-a%7D%2C%5Cquad+a%3Cx%3Cb%5C%5C+0%2C%5Cquad%5Cquad%5Cquad+%E5%85%B6%E4%BB%96%5Cend%7Bcases%7D)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=%28a%2Cb%29)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=X%5Csim+U%28a%2Cb%29)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=F%28x%29%3D%5Cbegin%7Bcases%7D+0%2C%5Cquad+%5Cquad+%5C%2Cx%3Ca%2C%5C%5C+%5Cfrac%7Bx-a%7D%7Bb-a%7D%2C%5Cquad+a%5Cleq+x%3Cb%2C%5C%5C+1%2C%5Cquad+%5Cquad+%5C%2Cx%5Cgeq+b.++%5Cend%7Bcases%7D)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=%5Cfrac%7Ba%2Bb%7D%7B2%7D)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=%5Cfrac%7B%28b-a%29%5E2%7D%7B12%7D)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 指数分布
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=p%28x%29%3D%5Cbegin%7Bcases%7D+%5Clambda%7B%5Crm+e%7D%5E%7B-%5Clambda+x%7D%2C%5Cquad+x%5Cgeq0%5C%5C+%5Cquad+0%2C%5Cquad+%5C%2C%5C%2C%5C%2Cx%3C0+%5Cend%7Bcases%7D)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=X%5Csim++Exp%28%5Clambda%29)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=F%28x%29%3D%5Cbegin%7Bcases%7D+1-%7B%5Crm+e%7D%5E%7B%5Clambda+x%7D%2C%5Cquad+x%5Cgeq0%5C%5C+%5Cquad+%5C+0%2C%5Cquad%5Cquad+%5C%2C%5C%2C+x%3C0+%5Cend%7Bcases%7D)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=%5Cfrac%7B1%7D%5Clambda)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=%5Cfrac%7B1%7D%7B%5Clambda%5E2%7D)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 伽玛分布
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=%5CGamma+%28+%5Calpha+%29+%3D+%5Cint+_+%7B+0+%7D+%5E+%7B+%5Cinfty+%7D+x+%5E+%7B+%5Calpha+-+1+%7D+e+%5E+%7B+-+x+%7D+d+x)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=%5Calpha%3E0)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=%5CGamma%281%29%3D1%2C%5Cquad+%5CGamma%28%5Cfrac%7B1%7D%7B2%7D%29%3D%5Csqrt+%7B%5Cpi%7D)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=%5CGamma%28%5Calpha%2B1%29%3D%5Calpha%5CGamma%28%5Calpha%29)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=%5Calpha)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=%5CGamma%28n%2B1%29%3Dn%5CGamma%28n%29%3Dn%5C%2C%21)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=p+%28+x+%29+%3D+%5Cleft%5C%7B+%5Cbegin%7Barray%7D+%7B+l+l+%7D+%7B+%5Cfrac+%7B+%5Clambda+%5E+%7B+a+%7D+%7D+%7B+%5CGamma+%28+%5Calpha+%29+%7D+x+%5E+%7B+a+-+1+%7D+%5Cmathrm+%7B+e+%7D+%5E+%7B+-+%5Clambda+x+%7D+%2C+%7D+%26+%7B+x+%5Cgeqslant+0%7D+%5C%5C+%7B+%5Cquad+%5Cquad+%5Cquad0%2C+++%7D+%26+%7B+x+%3C+0+%7D+%5Cend%7Barray%7D+%5Cright.)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=X%5Csim+Ga%28%5Calpha%2C%5Clambda%29)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=%5Calpha%3E0)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=%5Clambda%3E0)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=%5Cfrac%7B%5Calpha%7D%7B%5Clambda%7D)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=%5Cfrac%7B%5Calpha%7D%7B%5Clambda%5E2%7D)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 贝塔分布
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=%5Cmathrm+%7B+B+%7D+%28+a+%2C+b+%29+%3D+%5Cint+_+%7B+0+%7D+%5E+%7B+1+%7D+x+%5E+%7B+a+-+1+%7D+%28+1+-+x+%29+%5E+%7B+b+-+1+%7D+d+x)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=a%3E0%2Cb%3E0)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=B%28a%2Cb%29%3DB%28b%2Ca%29)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=%5Cmathrm+%7B+B+%7D+%28+a+%2C+b+%29+%3D+%5Cfrac+%7B+%5CGamma+%28+a+%29+%5C%2C%5CGamma+%28+b+%29+%7D+%7B+%5CGamma+%28+a+%2B+b+%29+%7D)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=p+%28+x+%29+%3D+%5Cleft%5C%7B+%5Cbegin%7Barray%7D+%7B+l+l+%7D+%7B%5Cfrac+%7B+%5CGamma+%28+a%2Bb+%29++%7D+%7B+%5CGamma+%28+a++%29+%5CGamma+%28+b+%29%7D+x+%5E+%7B+a+-+1+%7D+%281-x%29%5E%7Bb-1%7D+%7D%2C+%26+%7B0%3Cx%3C1%7D+%5C%5C+%7B+%5Cquad+%5Cquad+%5Cquad0%2C+++%7D+%26+%7B+%5Cquad+%E5%85%B6%E4%BB%96+%7D+%5Cend%7Barray%7D+%5Cright.)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=X%5Csim+Be%28a%2Cb%29)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=%5Cfrac%7Ba%7D%7Ba%2Bb%7D)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=%5Cfrac%7Bab%7D%7B%28a%2Bb%29%5E2%28a%2Bb%2B1%29%7D)

# COMMAND ----------

# MAGIC %md
# MAGIC # 条件概率
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=P%28A%7CB%29)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=P%28A%7CB%29+%3D+%5Cfrac%7BP%28AB%29%7D%7BP%28B%29%7D)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=P%28AB%29%3DP%28A%7CB%29P%28B%29%3DP%28B%7CA%29P%28A%29)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=B_1)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=B_2)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=B_1,B_2,%5Ccdots)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=B_i)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=B_i%5Ccap+B_j%3D%5Cphi)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=i%5Cne+j)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=i,j%3D1,2,%5Ccdots)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=P%28B_i%29%5Cgt0)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=i%3D1,2,%5Ccdots)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=B_1%2BB_2%5Ccdots%3D%5COmega)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=%5COmega)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=P%28A%29%3DP%28B_1%29P%28A%7CB_1%29%2BP%28B_2%29P%28A%7CB_2%29%2B...%2BP%28B_n%29P%28A%7CB_n%29%3D%5Csum_%7Bi%3D1%7D%5En+P%28B_i%29P%28A%7CB_i%29)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=P%28A%29%3DP%28B_1%29P%28A%7CB_1%29%2BP%28B_2%29P%28A%7CB_2%29%2B...%2BP%28B_n%29P%28A%7CB_n%29)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=%5Cne)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=A%5Ccap+B)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=B_iB_j%3D%5Cphi)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=B_1%2BB_2%2BB_3...%2BB_n%3D%5COmega)

# COMMAND ----------

# MAGIC %md
# MAGIC # 贝叶斯公式
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=B_i)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=P%28A%29%5Cgt0)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=P%28B_i%7CA%29%3D%5Cfrac%7BP%28A%7CB_i%29P%28B_i%29%7D%7BP%28A%29%7D%3D%5Cfrac%7BP%28A%7CB_i%29P%28B_i%29%7D%7B%5Csum_%7Bj%3D1%7D%5En+P%28B_j%29P%28A%7CB_j%29%7D)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=P%28B_i%29)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=P%28B_i%7CA%29)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 朴素贝叶斯

# COMMAND ----------

# MAGIC %md
# MAGIC ### 朴素贝叶斯分类
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=P%28类别%7C特征%29%3D%5Cfrac%7BP%28特征%7C类别%29P%28类别%29%7D%7BP%28特征%29%7D)
