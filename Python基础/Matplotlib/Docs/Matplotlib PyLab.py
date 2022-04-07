# Databricks notebook source
# MAGIC %md
# MAGIC # Matplotlib PyLab
# MAGIC 
# MAGIC PyLab 是一个面向 Matplotlib 的绘图库接口，其语法和 MATLAB 十分相近。它和 Pyplot 模快都够实现 Matplotlib 的绘图功能。
# MAGIC 
# MAGIC PyLab 是一个单独的模块，随 Matplotlib 软件包一起安装，该模块的导包方式和 Pyplot 不同。

# COMMAND ----------

import pylab
# 或者
from pylab import *

# COMMAND ----------

# MAGIC %md
# MAGIC 看一个简单的例子，画出正弦、余弦图形

# COMMAND ----------

from pylab import *
import numpy as np
import math

x = np.arange(0, math.pi*2, 0.05)
y = np.sin(x)

plot(x, np.sin(x))
plot(x, np.cos(x))
show()

# COMMAND ----------

# MAGIC %md
# MAGIC 如要绘制符号而不是线，请提供其他字符串参数，可用的符号参数如下：
# MAGIC 
# MAGIC * 符号：'-'，'--'，'-.'，':'，'.'，','，，o，^，v，<，>，s，+，x，D，d，1，2，3，4，h，H，p，| ，_
# MAGIC * 颜色：b(蓝色)，g(绿色)，r(红色)，c(青色)，m(品红)，y(黄色)，k(黑色)，w(白色)

# COMMAND ----------

from pylab import *

x = linspace(-10, 10, 20)

plot(x, x, 'r.')
plot(x, x**2, 'g--')
plot(x, -x**2, 'b^')
show()

# COMMAND ----------

from pylab import *
import numpy as np
import math

x = np.arange(0, math.pi*2, 0.05)

plot(x, sin(x))
plot(x, cos(x), 'r-')
plot(x, -sin(x), 'g--')
show()
