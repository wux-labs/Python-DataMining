# Databricks notebook source
# MAGIC %md
# MAGIC # 希腊字母表
# MAGIC 
# MAGIC | 序号 | 大写 | 小写 | 英文注音 | 国际音标注音 | 中文读音 | 意义 |
# MAGIC | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
# MAGIC | 1 | Α | α | alpha | /ˈælfə/ | 阿尔法 | 角度；系数 |
# MAGIC | 2 | Β | β | beta | /'beitə/ | 贝塔 | 磁通系数；角度；系数 |
# MAGIC | 3 | Γ | γ | gamma | /'gæmə/ | 伽马 | 电导系数（小写） |
# MAGIC | 4 | Δ | δ | delta | /'deltə/ | 德尔塔 | 变动；密度；屈光度 |
# MAGIC | 5 | Ε | ε | epsilon | /ep'silɒn/ | 伊普西龙 | 对数之基数 |
# MAGIC | 6 | Ζ | ζ | zeta | /'zi:tə/ | 截塔 | 系数；方位角；阻抗；相对粘度；原子序数 |
# MAGIC | 7 | Η | η | eta | /'i:tə/ | 艾塔 | 磁滞系数；效率（小写） |
# MAGIC | 8 | Θ | θ | theta | /'θi:tə/ | 西塔 | 温度；相位角 |
# MAGIC | 9 | Ι | ι | iota | /ai'əute/ | 约塔 | 微小，一点儿 |
# MAGIC | 10 | Κ | κ | kappa | /'kæpə/ | 卡帕 | 介质常数 |
# MAGIC | 11 | Λ | λ | lambda | /'læmdə/ | 兰布达 | 波长（小写）；体积 |
# MAGIC | 12 | Μ | μ | mu | /mju:/ | 缪 | 磁导系数微（千分之一）放大因数（小写） |
# MAGIC | 13 | Ν | ν | nu | /nju:/ | 纽 | 磁阻系数 |
# MAGIC | 14 | Ξ | ξ | xi | /ksi/ | 克西 | 数学上的随机变量 |
# MAGIC | 15 | Ο | ο | omicron | /əumaik'rɒn/ | 奥密克戎 | 高阶无穷小函数 |
# MAGIC | 16 | Π | π | pi | /pai/ | 派 | 圆周率=圆周÷直径=3.14159 26535 89793 |
# MAGIC | 17 | Ρ | ρ | rho | /rou/ | 肉 | 电阻系数（小写）、密度（小写） |
# MAGIC | 18 | Σ | σ | sigma | /'sigmə/ | 西格马 | 总和（大写），表面密度；跨导（小写） |
# MAGIC | 19 | Τ | τ | tau | /tau/ | 套 | 时间常数 |
# MAGIC | 20 | Υ | υ | upsilon | /ʌp'sailən/ | 艾普西龙 | 位移 |
# MAGIC | 21 | Φ | φ | phi | /fai/ | 佛爱 | 磁通；角 |
# MAGIC | 22 | Χ | χ | chi | /kai/ | 西 | 统计学中有卡方(χ2)分布 |
# MAGIC | 23 | Ψ | ψ | psi | /psai/ | 普塞 | 角速；介质电通量（静电力线）；角 |
# MAGIC | 24 | Ω | ω | omega | /'əumigə/ | 欧米伽 | 欧姆（大写）；角速（小写）；角 |

# COMMAND ----------

# MAGIC %md
# MAGIC # 单个符号
# MAGIC 
# MAGIC https://www.tsfhdq.cn/mfjc3/1132.html
# MAGIC 
# MAGIC https://www.zhihu.com/question/355715459
# MAGIC 
# MAGIC https://zhuanlan.zhihu.com/p/392807096
# MAGIC 
# MAGIC https://zhuanlan.zhihu.com/p/94181395
# MAGIC 
# MAGIC 
# MAGIC `%28` = ![](https://www.zhihu.com/equation?tex=%28)
# MAGIC 
# MAGIC `%29` = ![](https://www.zhihu.com/equation?tex=%29)
# MAGIC 
# MAGIC `%2B` = ![](https://www.zhihu.com/equation?tex=%2B)
# MAGIC 
# MAGIC `B^T` = ![](https://www.zhihu.com/equation?tex=B^T)
# MAGIC 
# MAGIC `x_0` = ![](https://www.zhihu.com/equation?tex=x_0)
# MAGIC 
# MAGIC `%5Cdelta` = ![](https://www.zhihu.com/equation?tex=%5Cdelta)
# MAGIC 
# MAGIC `%5CDelta` = ![](https://www.zhihu.com/equation?tex=%5CDelta)
# MAGIC 
# MAGIC `%5Crightarrow` = ![](https://www.zhihu.com/equation?tex=%5Crightarrow)
# MAGIC 
# MAGIC `%C2%B1` = ![](https://www.zhihu.com/equation?tex=%C2%B1)
# MAGIC 
# MAGIC `%5CDelta+x` = ![](https://www.zhihu.com/equation?tex=%5CDelta+x)
# MAGIC 
# MAGIC `%5Clambda` = ![](https://www.zhihu.com/equation?tex=%5Clambda)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=f%27%28x_0%29%3D%5Clim_%7B%5CDelta+x%5Crightarrow+0%7D)
# MAGIC 
# MAGIC  ≈ ≡ ≠ ＝ ≤≥ ＜ ＞ ≮ ≯ ∷ ± ＋ － × ÷ ／ ∫ ∮ ∝ ∞ ∧ ∨ ∑ ∏ ∪ ∩ ∈ ∵ ∴  ⊥ ‖ ∠ ⌒  ≌ ∽ √  （） 【】｛｝ Ⅰ Ⅱ ⊕ ⊙∥α β γ δ ε ζ η θ Δ

# COMMAND ----------

# MAGIC %md
# MAGIC # 数学符号
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=%5Cvec%7Ba%7D)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=%5Cvec%7Bx%7D)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=A%5Cvec%7Bx%7D)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=%5Cvec%7By%7D=A%5Cvec%7Bx%7D)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=A^T)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=%28A^T%29^T=A)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=%28λA%29^T=λA^T)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=%28AB%29^T=B^TA^T)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=%28A%2BB%29^T=A^T%2BB^T)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=x_0)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=y=f%28x%29)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=U%28x_0%29)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=%5Cdelta)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=x_0%2BΔx∈U%28x_0%29)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=Δy=f%28x_0%2BΔx%29-f%28x_0%29)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=Δy)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=Δx→0)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=f'%28x_0%29)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=f%28x%29)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=f'%28x%29)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=x^n)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=e^x)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=a^x)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=a^x%5Cln+a)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=%5Cln+x)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=%5Csin+x)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=-%5Csin+x)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=%5Ccos+x)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=nx^%7Bn-1%7D)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=%5Cfrac%7B1%7D%7Bx%7D)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=-%5Cfrac%7B1%7D%7Bx^2%7D)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=X^T%3DX)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=AA^T=A^TA=E)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=f'%28x_0%29%3D%5Csum_%7Bi%3D1%7D%5En)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=f%27%28x%29%3D%5Clim_%7B%5CDelta+x%5Crightarrow+0%7D%5Cfrac%7Bf%28x%2B%5CDelta+x%29-f%28x%29%7D%7B%5CDelta+x%7D%3D%5Clim+_%7B%5CDelta+x%5Crightarrow+0%7D%5Cfrac%7B%28x%2B%5CDelta+x%29%5E2+-x%5E2%7D%7B%5CDelta+x%7D)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=f%27%28x_0%29%3D%5Clim_%7B%5CDelta+x%5Crightarrow+0%7D)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=%5Cfrac%7B%5CDelta+y%7D%7B%5CDelta+x%7D)
# MAGIC 
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=f%27%28x_0%29%3D%5Clim_%7B%5CDelta+x%5Crightarrow+0%7D%5Cfrac%7B%5CDelta+y%7D%7B%5CDelta+x%7D)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=%5Cfrac%7B%5CDelta+AX%7D%7B%5CDelta+X%7D%3DA^T)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=%5Cfrac%7B%5CDelta+X^TAX%7D%7B%5CDelta+X%7D%3D%28A%2BA^T%29X)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=f%27%28x_0%29%3D%5Clim_%7B%5CDelta+x%5Crightarrow+0%7D%5Cfrac%7B%5CDelta+y%7D%7B%5CDelta+x%7D%3D%5Cfrac%7Bf%28x_0%2B%5CDelta+x%29-f%28x_0%29%7D%7B%5CDelta+x%7D)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=f%27%28x_0%29%3D%5Clim_%7B%5CDelta+x%5Crightarrow+0%7D%5Cfrac%7B%5CDelta+y%7D%7B%5CDelta+x%7D%3D%5Cfrac%7Bf%28x_0%2B%5CDelta+x%29-f%28x_0%29%7D%7B%5CDelta+x%7D)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=f%27%28x_0%29%3D%5Clim_%7Bx%5Crightarrow+x_0%7D%5Cfrac%7Bf%28x%29-f%28x_0%29%7D%7Bx-x_0%7D)
# MAGIC 
# MAGIC 11
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=f%27%28x_0%29%3D%5Clim_%7B%5CDelta+x%5Crightarrow+0%7D%5Cfrac%7Bf%28x%2B%5CDelta+x%29-f%28x%29%7D%7B%5CDelta+x%7D%3D%5Clim+_%7B%5CDelta+x%5Crightarrow+0%7D%5Cfrac%7B%28x%2B%5CDelta+x%29%5E2+-x%5E2%7D%7B%5CDelta+x%7D)
# MAGIC 
# MAGIC 22
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=%5Cnabla_%7B%5Cpmb%7Bx%7D%7Df%28%5Cpmb%7Bx%7D%29%3D+%5Cfrac%7B%5Cpartial+f%28%5Cpmb%7Bx%7D%29%7D%7B%5Cpartial+%5Cpmb%7Bx%7D%7D%3D++%5Cleft%5B++%5Cfrac%7B%5Cpartial+f%7D%7B%5Cpartial+x_1%7D%2C++%5Cfrac%7B%5Cpartial+f%7D%7B%5Cpartial+x_2%7D%2C++%5Ccdots%2C+%5Cfrac%7B%5Cpartial+f%7D%7B%5Cpartial+x_n%7D++%5Cright%5D%5ET+%5C%5C%5C%5C++%5Ctag%7B%E6%9C%AC%E8%B4%A8%E7%AF%87_6%7D)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=df%3D%5Csum_%7Bi%3D1%7D%5En+%5Cfrac%7B%5Cpartial+f%7D%7B%5Cpartial+x_i%7Ddx_i%3D%5Cfrac%7B%7B%5Cpartial+f%7D%5ET%7D%7B%5Cpartial+x%7Ddx)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=E%3D%5Csum_%7Bi%3D1%7D%5En%28y-y_i%29^2)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=j%28θ%29)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=y%3Dh_θ%28x^%7B%28i%29%7D%29%3Dxθ)
# MAGIC 
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=%5Cbegin%7Balign%7D+%5Cfrac%7B%5Cpartial%7Bc%7D%7D%7B%5Cpartial%7B%5Cpmb%7BX%7D%7D%7D+%26%3D+%5Cbegin%7Bbmatrix%7D+%5Cfrac%7B%5Cpartial%7Bc%7D%7D%7B%5Cpartial%7Bx_%7B11%7D%7D%7D%26%5Cfrac%7B%5Cpartial%7Bc%7D%7D%7B%5Cpartial%7Bx_%7B12%7D%7D%7D%26%5Ccdots%26%5Cfrac%7B%5Cpartial%7Bc%7D%7D%7B%5Cpartial%7Bx_%7B1n%7D%7D%7D+%5C%5C++%5Cfrac%7B%5Cpartial%7Bc%7D%7D%7B%5Cpartial%7Bx_%7B21%7D%7D%7D%26%5Cfrac%7B%5Cpartial%7Bc%7D%7D%7B%5Cpartial%7Bx_%7B22%7D%7D%7D%26%5Ccdots%26%5Cfrac%7B%5Cpartial%7Bc%7D%7D%7B%5Cpartial%7Bx_%7B2n%7D%7D%7D++%5C%5C+%5Cvdots+%26%5Cvdots+%26+%5Cvdots+%26+%5Cvdots%5C%5C+%5Cfrac%7B%5Cpartial%7Bc%7D%7D%7B%5Cpartial%7Bx_%7Bm1%7D%7D%7D%26%5Cfrac%7B%5Cpartial%7Bc%7D%7D%7B%5Cpartial%7Bx_%7Bm2%7D%7D%7D%26%5Ccdots%26%5Cfrac%7B%5Cpartial%7Bc%7D%7D%7B%5Cpartial%7Bx_%7Bmn%7D%7D%7D++%5Cend%7Bbmatrix%7D_%7Bm+%5Ctimes+n%7D+%5C%5C%5C%5C+%26%3D++%5Cbegin%7Bbmatrix%7D++0%260%26%5Ccdots%260+%5C%5C++0%260%26%5Ccdots%260+%5C%5C+%5Cvdots+%26%5Cvdots+%26+%5Cvdots+%26+%5Cvdots%5C%5C+0%260%26%5Ccdots%260++%5Cend%7Bbmatrix%7D_%7Bm+%5Ctimes+n%7D+%5C%5C%5C%5C++%26%3D%5Cpmb%7B0%7D_%7Bm+%5Ctimes+n%7D%5Cend%7Balign%7D+%5C%5C%5C%5C++%5Ctag%7B19%7D)
# MAGIC 
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=%5Cbegin%7Balign%7D+%5Cbegin%7Bbmatrix%7D+%5Cfrac%7B%5Cpartial%7Bc%7D%7D%7B%5Cpartial%7Bx_%7B11%7D%7D%7D%26%5Cfrac%7B%5Cpartial%7Bc%7D%7D%7B%5Cpartial%7Bx_%7B12%7D%7D%7D%26%5Ccdots%26%5Cfrac%7B%5Cpartial%7Bc%7D%7D%7B%5Cpartial%7Bx_%7B1n%7D%7D%7D+%5C%5C++%5Cfrac%7B%5Cpartial%7Bc%7D%7D%7B%5Cpartial%7Bx_%7B21%7D%7D%7D%26%5Cfrac%7B%5Cpartial%7Bc%7D%7D%7B%5Cpartial%7Bx_%7B22%7D%7D%7D%26%5Ccdots%26%5Cfrac%7B%5Cpartial%7Bc%7D%7D%7B%5Cpartial%7Bx_%7B2n%7D%7D%7D++%5C%5C+%5Cvdots+%26%5Cvdots+%26+%5Cvdots+%26+%5Cvdots%5C%5C+%5Cfrac%7B%5Cpartial%7Bc%7D%7D%7B%5Cpartial%7Bx_%7Bm1%7D%7D%7D%26%5Cfrac%7B%5Cpartial%7Bc%7D%7D%7B%5Cpartial%7Bx_%7Bm2%7D%7D%7D%26%5Ccdots%26%5Cfrac%7B%5Cpartial%7Bc%7D%7D%7B%5Cpartial%7Bx_%7Bmn%7D%7D%7D++%5Cend%7Bbmatrix%7D_%7Bm+%5Ctimes+n%7D+%5C%5C%5C%5C+%26%5Cend%7Balign%7D)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=%5Cbegin%7Bbmatrix%7D++0%260%26%5Ccdots%260+%5C%5C++0%260%26%5Ccdots%260+%5C%5C+%5Cvdots+%26%5Cvdots+%26+%5Cvdots+%26+%5Cvdots%5C%5C+0%260%26%5Ccdots%260++%5Cend%7Bbmatrix%7D)
# MAGIC 
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=f%28x%29%3Dx_1^2%2Bx_2^2%2b0.1x_1x_2)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=%5Cbegin%7Bbmatrix%7D++1%260.05+%5C%5C++0.05%261++%5Cend%7Bbmatrix%7D)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=f%28x%29%3Dx_1^2%2Bx_2^2%2b2x_1x_2)
# MAGIC 
# MAGIC ![](https://www.zhihu.com/equation?tex=%5Cbegin%7Bbmatrix%7D++1%261+%5C%5C++1%261++%5Cend%7Bbmatrix%7D)
