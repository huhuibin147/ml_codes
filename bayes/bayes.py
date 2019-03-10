# -*- coding: utf-8 -*-

import numpy as np
import jieba


data = []
cutdata = []
labels = []

with open('data.txt', encoding='utf8') as f:
    doc = f.read()
    docs = doc.split('\n')
    for s in docs:
        sent, label = s.split('|')
        data.append(sent)
        cutdata.append(jieba.lcut(sent))
        labels.append(int(label))
    

vocabset = set([])
for s in cutdata:
    vocabset = vocabset | set(s)
vocab = list(vocabset)

trainMat = []

for r in cutdata:
    wordVec = [0] * len(vocab)
    for w in r:
        wordVec[vocab.index(w)] += 1
    trainMat.append(wordVec)
    

def train(trainM, labels):
    numDocs = len(trainM)
    numWords = len(trainM[0])
    pA = sum(labels) / numDocs
    p0Count = np.ones(numWords)
    p1Count = np.ones(numWords)
    p0Ws = 2
    p1Ws = 2
    for i in range(numDocs):
        if labels[i] == 1:
            p1Count += trainM[i]
            p1Ws += sum(trainM[i])
        else:
            p0Count += trainM[i]
            p0Ws += sum(trainM[i])
    p1Vec = np.log(p1Count/p1Ws)
    p0Vec = np.log(p0Count/p0Ws)
    return p0Vec, p1Vec, pA

p0Vec, p1Vec, pA = train(trainMat, labels)

def classify(wv):
    p1 = sum(wv * p1Vec) + np.log(pA)
    p0 = sum(wv * p0Vec) + np.log(1 - pA)
    if p1 > p0:
        return 1
    else:
        return 0
    

def s2doc(seqs):
    wordVec = [0] * len(vocab)
    for w in jieba.lcut(seqs):
        wordVec[vocab.index(w)] += 1
    return wordVec
    

def test(seqs):
    wv = s2doc(seqs)
    c = classify(wv)
    print('分类结果:%s' % c)

test('啊咧')


