# Databricks notebook source
# MAGIC %md
# MAGIC 初始化脚本安装决策树画图类库

# COMMAND ----------

dbutils.fs.mkdirs("dbfs:/databricks/scripts/")

dbutils.fs.put("dbfs:/databricks/scripts/initialization.sh",
"""
#!/bin/bash

# install dependent packages
# sudo apt-get install -y python3-dev graphviz libgraphviz-dev pkg-config
# sudo pip install graphviz pygraphviz

""", True)

# COMMAND ----------

dbutils.fs.ls("dbfs:/databricks/scripts")

# COMMAND ----------

# MAGIC %md
# MAGIC 解决matplotlib中文乱码问题

# COMMAND ----------

import matplotlib as mpl

mpl.get_cachedir()

mpl.matplotlib_fname()

# COMMAND ----------

# MAGIC %sh
# MAGIC 
# MAGIC cp SimHei.ttf /usr/share/fonts/
# MAGIC ls -al /usr/share/fonts/
# MAGIC 
# MAGIC rm -rf /root/.cache/matplotlib/*
# MAGIC 
# MAGIC echo "font.family : sans-serif" >> /databricks/python/lib/python3.8/site-packages/matplotlib/mpl-data/matplotlibrc
# MAGIC echo "font.sans-serif : SimHei" >> /databricks/python/lib/python3.8/site-packages/matplotlib/mpl-data/matplotlibrc
# MAGIC echo "axes.unicode_minus : False" >> /databricks/python/lib/python3.8/site-packages/matplotlib/mpl-data/matplotlibrc
# MAGIC 
# MAGIC cat /databricks/python/lib/python3.8/site-packages/matplotlib/mpl-data/matplotlibrc
