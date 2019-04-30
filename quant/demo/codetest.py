# coding: utf-8

import pandas as pd
import numpy as np



# pandas类似numpy一维数组结构
# data可以是任意对象,index是可以省略的
idx = ["one", "two", "three"]
data = [1, 2, 3]
d1 = pd.Series(data, idx)
print('[d1]\n', d1)

# 使用numpy类型
d2 = pd.Series(np.array(data))
print('[d2]\n', d2)

# 使用dict类型
dd1 = {"a":"1", "b":"2", "c":"3"}
d3 = pd.Series(dd1)
print('[d3]\n', d3)

# 索引计算,不存在NaN
d4 = pd.Series({"a":"1", "b":"2", "d":"3"})
d5 = d3 + d4
print('[d5]\n', d5)

# DataFrame  2维数据结构,由series组成
data = {
    'name': ['aa', 'oo', 'ee'],
    'age': [20, 12, 30],
    'year': [2001, 1993, 1991]
}
df = pd.DataFrame(data, index=['q1', 'q2', 'q3'])
print('[df]\n', df)
print('[df->name]\n', df['name'])


# 处理
# d = pd.read_csv("blb.csv")
# print(d)

