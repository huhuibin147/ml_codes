# coding: utf-8

import pandas as pd
import numpy as np



# pandas类似numpy一维数组结构
# data可以是任意对象
idx = ["one", "two", "three"]
data = [1, 2, 3]
d1 = pd.Series(data, idx)
print(d1)






# 处理练习
# d = pd.read_csv("blb.csv")
# print(d)

