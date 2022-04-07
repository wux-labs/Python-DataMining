# Databricks notebook source
# MAGIC %md
# MAGIC # Seaborn
# MAGIC Seaborn 是一个基于 matplotlib 且数据结构与 pandas 统一的统计图制作库。
# MAGIC 
# MAGIC Seaborn 框架旨在以数据可视化为中心来挖掘与理解数据。它提供的面向数据集制图函数主要是对行列索引和数组的操作，包含对整个数据集进行内部的语义映射与统计整合，以此生成富于信息的图表。

# COMMAND ----------

import seaborn as sns

# 装载数据集
tips = sns.load_dataset("tips")

sns.relplot(x="total_bill", y="tip", col="time",
            hue="smoker", style="smoker", size="size",
            data=tips);

# COMMAND ----------

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def sinplot(flip=1):
    x = np.linspace(1,14,100)
    for i in range(1,7):
        plt.plot(x, np.sin(x + i * 0.5) * (7 - i) * flip)

sinplot()
