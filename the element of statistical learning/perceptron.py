# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# 数据集处理
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['label'] = iris.target
df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']

# label中各分类统计
df.label.value_counts()

# 数据集可视化，展示label 0,1
def show_data():
    plt.scatter(df[:50]['sepal length'], df[:50]['sepal width'], label='0')
    plt.scatter(df[50:100]['sepal length'], df[50:100]['sepal width'], label='1')
    plt.xlabel('sepal length')
    plt.ylabel('sepal width')
    plt.legend()

show_data()

# 数据处理
data = np.array(df.iloc[:100, [0,1,-1]])
X, y = data[:,:-1], data[:,-1]
y = np.array([1 if i == 1 else -1 for i in y])


# 初始化参数
w = np.ones(len(data[0])-1, dtype=np.float32)
b = 0
l_rate = 0.1

def sign(x, w, b):
    return np.dot(x,w)+b

def fit(x_, y_):
    global w, b
    is_stop = False
    while not is_stop:
#        print('[epoch] ',w,b)
        wrong_count = 0
        for d in range(len(x_)):
            x = x_[d]
            y = y_[d]
            if y*sign(x,w,b) <= 0:
#                print('x,y:',x,y)
                w=w+l_rate*np.dot(y,x)
                b=b+l_rate*y
                wrong_count+=1
        if wrong_count == 0:
            is_stop = True

fit(X, y)

x_points=np.linspace(4,7,10)
# wx+b=0 这里的y是对应x2，换了位置
y_=-(w[0]*x_points+b)/w[1]
plt.plot(x_points,y_)
plt.plot(data[:50,0], data[:50,1], 'bo', color='blue', label='0')
plt.plot(data[50:100,0], data[50:100,1], 'bo', color='orange', label='1')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend()

# =============================================================================
# scikit-learn preceptron 结果会有点不一样
# =============================================================================
from sklearn.linear_model import Perceptron
clf = Perceptron(fit_intercept=False, max_iter=1000, shuffle=False)
clf.fit(X, y)
print(clf.coef_, clf.intercept_)
x_ponits = np.arange(4, 8)
y_ = -(clf.coef_[0][0]*x_ponits + clf.intercept_)/clf.coef_[0][1]
plt.plot(x_ponits, y_)

plt.plot(data[:50, 0], data[:50, 1], 'bo', color='blue', label='0')
plt.plot(data[50:100, 0], data[50:100, 1], 'bo', color='orange', label='1')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend()