# -*- coding: utf-8 -*-
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from collections import Counter

# 距离公式
def L(x, y, p=1):
    sum = 0
    for i in range(len(x)):
        sum += math.pow(abs(x[i] - y[i]), p)
    return math.pow(sum, 1/p)    
    
L([1,1], [5,1], 2)


iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['label'] = iris.target
df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']


plt.scatter(df[:50]['sepal length'], df[:50]['sepal width'], label='0')
plt.scatter(df[50:100]['sepal length'], df[50:100]['sepal width'], label='1')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend()

data = np.array(df.iloc[:100, [0,1,-1]])
X, y = data[:,:-1], data[:,-1]
# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


n_neighbors = 3
p = 2

def predict(X):
    knn_list = []
    for i in range(n_neighbors):
        # np范式计算函数
        dist = np.linalg.norm(X-X_train[i], ord=p)
        knn_list.append((dist, y_train[i]))
    
    for i in range(n_neighbors, len(X_train)):
        max_index = knn_list.index(max(knn_list, key=lambda x:x[0]))
        dist = np.linalg.norm(X-X_train[i], ord=p)
        if dist < knn_list[max_index][0]:
            knn_list[max_index] = (dist, y_train[i])
    
    # 统计最大分类数
    knn = [k[-1] for k in knn_list]
    max_count = sorted(Counter(knn), key=lambda x:x)[-1]
    return max_count
    

#c = Counter([1,1,1,12,2,2])
#sorted(c, key=lambda x:x)

def score(X_test, y_test):
    right_count = 0
    for X, y in zip(X_test, y_test):
        if predict(X) == y:
            right_count += 1
    return right_count / len(X_test)

score(X_test, y_test)

test_point = [6.0, 3.0]
predict(test_point)


plt.scatter(df[:50]['sepal length'], df[:50]['sepal width'], label='0')
plt.scatter(df[50:100]['sepal length'], df[50:100]['sepal width'], label='1')
plt.plot(test_point[0], test_point[1], 'bo', label='test_point')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend()



# =============================================================================
# scikitlear实现
# =============================================================================
from sklearn.neighbors import KNeighborsClassifier

clf_sk = KNeighborsClassifier()
clf_sk.fit(X_train, y_train)
clf_sk.score(X_test, y_test)






