# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from collections import Counter
import math

def create_data():
    datasets = [['青年', '否', '否', '一般', '否'],
               ['青年', '否', '否', '好', '否'],
               ['青年', '是', '否', '好', '是'],
               ['青年', '是', '是', '一般', '是'],
               ['青年', '否', '否', '一般', '否'],
               ['中年', '否', '否', '一般', '否'],
               ['中年', '否', '否', '好', '否'],
               ['中年', '是', '是', '好', '是'],
               ['中年', '否', '是', '非常好', '是'],
               ['中年', '否', '是', '非常好', '是'],
               ['老年', '否', '是', '非常好', '是'],
               ['老年', '否', '是', '好', '是'],
               ['老年', '是', '否', '好', '是'],
               ['老年', '是', '否', '非常好', '是'],
               ['老年', '否', '否', '一般', '否'],
               ]
    labels = [u'年龄', u'有工作', u'有自己的房子', u'信贷情况', u'类别']
    # 返回数据集和每个维度的名称
    return datasets, labels

datasets, labels = create_data()
train_data = pd.DataFrame(datasets, columns=labels)


# 熵计算
def calc_ent(datasets):
    data_length = len(datasets)
    label_count = {}
    for i in range(data_length):
        label = datasets[i][-1]
        if label not in label_count:
            label_count[label] = 0
        label_count[label] += 1
    ent = -sum([(c/data_length)*math.log2(c/data_length) for c in label_count.values()])
    return ent

#calc_ent(np.array(datasets))

# 条件熵
def cond_ent(datasets, axis=0):
    data_length = len(datasets)
    feature_sets = {}
    for i in range(data_length):
        feature = datasets[i][axis]
        if feature not in feature_sets:
            feature_sets[feature] = []
        feature_sets[feature].append(datasets[i])
    cond_ent = sum([(len(p)/data_length)*calc_ent(p) for p in feature_sets.values()])
    return cond_ent

#cond_ent(np.array(datasets), axis=0)

# 信息增益
def info_gain(ent, cond_ent):
    return ent - cond_ent

# 特征选择
def info_gain_train(datasets):
    # 特征数
    count = len(datasets[0]) - 1
    ent = calc_ent(datasets)
    features_r = []
    for c in range(count):
        c_ent = cond_ent(datasets, c)
        c_i_g = info_gain(ent, c_ent)
        features_r.append((c, c_i_g))
        print('特征:{}, 信息增益:{:.3f}'.format(labels[c], c_i_g))
    print(features_r)
    best_f = max(features_r, key=lambda x:x[-1])
    return best_f

info_gain_train(np.array(datasets))

class Node:
    def __init__(self, root=True, label=None, feature_name=None, feature=None):
        self.root = root
        self.label = label
        self.feature_name = feature_name
        self.feature = feature
        self.tree = {}
        self.result = {'label:': self.label, 'feature': self.feature, 'tree': self.tree}
    
    def add_node(self, val, node):
        self.tree[val] = node
        
    def __repr__(self):
        return '{}'.format(self.result)

class DTree:
    def __init__(self, epsilon=0.1):
        self.epsilon = epsilon
        self._tree = {}
        
    def train(self, train_data):
        _, y_train, features = train_data.iloc[:,:-1], train_data.iloc[:,-1], train_data.columns[:-1]
        if len(y_train.value_counts()) == 1:
            return Node(root=True, label=y_train.iloc[0])
        if len(features) == 0:
            return Node(root=True, label=y_train.value_counts().sort_values(ascending=False).index[0])
        max_feature, max_info_gain = info_gain_train(np.array(train_data))
        max_feature_name = features[max_feature]
        
        if max_info_gain < self.epsilon:
            # 按类数量多的当做label
            return Node(root=True, label=y_train.value_counts().sort_values(ascending=False).index[0])
        
        node_tree = Node(root=False, feature_name=max_feature_name, feature=max_feature)
        
        features_list = train_data[max_feature_name].value_counts().index
        for f in features_list:
            sub_train_df = train_data.loc[train_data[max_feature_name]==f].drop([max_feature_name], axis=1)
            sub_tree = self.train(sub_train_df)
            node_tree.add_node(f, sub_tree)
        
        return node_tree
        
    def fit(self, train_data):
        self._tree = self.train(train_data)
        return self._tree
    

dt = DTree()
tree = dt.fit(train_data)
print(tree)


# =============================================================================
# sklearn
# =============================================================================

def create_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target
    df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
    data = np.array(df.iloc[:100, [0, 1, -1]])
    # print(data)
    return data[:,:2], data[:,-1]

X, y = create_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

from sklearn.tree import DecisionTreeClassifier

from sklearn.tree import export_graphviz
import graphviz

clf = DecisionTreeClassifier()
clf.fit(X_train, y_train,)
clf.score(X_test, y_test)
tree_pic = export_graphviz(clf, out_file="mytree.pdf")
with open('mytree.pdf') as f:
    dot_graph = f.read()
graphviz.Source(dot_graph)