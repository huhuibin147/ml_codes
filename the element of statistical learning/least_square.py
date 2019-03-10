# -*- coding: utf-8 -*-

import numpy as np
from scipy.optimize import leastsq
import matplotlib.pyplot as plt

# 原函数
def real_func(x):
    return np.sin(2*np.pi*x)

# 多项式函数
def fit_func(p,x):
    f = np.poly1d(p)
    return f(x)

# 损失函数
def residuals_func(p,x,y):
    return fit_func(p,x) - y

x = np.linspace(0,1,10)
x_points = np.linspace(0,1,1000)

y_ = real_func(x)
y = [np.random.normal(0,0.1)+y1 for y1 in y_]

# 最小二乘法
def fitting(M=0):
    p_init=np.random.rand(M+1)
    p_lsq = leastsq(residuals_func,p_init,args=(x,y))
    print('Fitting Parameters:', p_lsq[0])

    plt.plot(x_points,real_func(x_points),label='real')
    plt.plot(x_points,fit_func(p_lsq[0],x_points),label='fitted curve')
    plt.plot(x,y,'bo',label='noise')
    plt.legend()
    return p_lsq


p_lsq0 = fitting(M=0)
p_lsq1 = fitting(M=1)
p_lsq3 = fitting(M=3)
p_lsq9 = fitting(M=9)

regularization = 0.00001

# 正则化
def residuals_func_regularization(p,x,y):
    ret = fit_func(p,x) - y
    ret = np.append(ret, np.sqrt(0.5*regularization*np.square(p)))
    return ret


def fitting2(M=0):
    p_init=np.random.rand(M+1)
    p_lsq = leastsq(residuals_func,p_init,args=(x,y))
    return p_lsq

def f2(M=9):
    p_init = np.random.rand(M+1)
    p_lsq_regularization = leastsq(residuals_func_regularization, p_init, args=(x,y))
    
    plt.plot(x_points,real_func(x_points), label='real')
    plt.plot(x_points,fit_func(fitting2(M)[0],x_points), label='fitted curver')
    plt.plot(x_points,fit_func(p_lsq_regularization[0],x_points),label='regularization')
    plt.plot(x,y,'bo',label='noise')
    plt.legend()


p9 = f2(9)




