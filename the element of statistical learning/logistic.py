# -*- coding: utf-8 -*-

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

def create_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target
    df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
    data = np.array(df.iloc[:100, [0,1,-1]])
    return data [:,:-1], data[:,-1]


X, y = create_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

def sigmoid(x):
    return 1 / (1 + math.exp(-x))


epoches = 200
lr = 0.01
def fit(X, y):
    new_X = data_matrix(X)
    # 初始化为列向量
    w = np.zeros((len(new_X[0]), 1), dtype=np.float32)
    for epoch in range(epoches):
        for i in range(len(X)):
            a = np.dot(new_X[i], w)
            error = sigmoid(a) - y[i]
            # 这里new_X[i]是list，np转置第一次是1维，需要先加1维
            w -= lr * error * np.transpose([new_X[i]])
    return w

#np.zeros((len(X[0]),1), dtype=np.float32)

# 初始化X0值为1，w0+w1x+w2x
def data_matrix(X):
    data_mat = []
    for d in X:
        data_mat.append([1.0, *d])
    return data_mat

#d=data_matrix(X)
#np.transpose([d[0]])

# list 对 array 可以做矩阵计算
#np.dot([1,3], np.array([[1],[2]]))
    
w = fit(X_train, y_train)

def score(X_test, y_test, w):
    right = 0
    X_test = data_matrix(X_test)
    for x,y in zip(X_test, y_test):
        res = np.dot(x, w)
        if (res > 0 and y == 1) or (res < 0 and y == 0):
            right += 1
    return right / len(X_test)

print(score(X_test, y_test, w))

x_points = np.arange(4, 8)
y_ = -(w[0] + w[1]*x_points) / w[2]
plt.plot(x_points, y_)
plt.scatter(X[:50,0],X[:50,1],label='0')
plt.scatter(X[50:100,0],X[50:100,1],label='1')
plt.legend()


# =============================================================================
# sklearn
# =============================================================================
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(max_iter=200)
clf.fit(X_train, y_train)
clf.score(X_test, y_test)
print(clf.coef_, clf.intercept_)
x_ponits = np.arange(4, 8)
y_ = -(clf.coef_[0][0]*x_ponits + clf.intercept_)/clf.coef_[0][1]
plt.plot(x_ponits, y_)

plt.plot(X[:50, 0], X[:50, 1], 'bo', color='blue', label='0')
plt.plot(X[50:, 0], X[50:, 1], 'bo', color='orange', label='1')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend()