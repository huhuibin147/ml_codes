# -*- coding: utf-8 -*-
import copy
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import os
os.chdir("D:/machinelearning/codes/ml_learning/NN/")
os.getcwd()



tf.reset_default_graph()


def init_weight(shape, name):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)

def init_bias(shape, name):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name)

def conv2d(x, W):
    # SAME算法 new_height = new_width = W / S （结果向上取整）
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')


x = tf.placeholder(tf.float32, [None, 784], name='x')
x_image = tf.reshape(x, [-1,28,28,1])

w_conv1 = init_weight([5,5,1,32], name='w1')
b_conv1 = init_bias([32], name='b1')

h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1)+b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# 第一层设置的32个核
w_conv2 = init_weight([5,5,32,64], name='w2')
b_conv2 = init_bias([64], name='b2')

h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2)+b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

w_fc1 = init_weight([7*7*64, 1024], name='wfc1')
b_fc1 = init_bias([1024], name='bfc1')

h_pool2_flag = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flag, w_fc1)+b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

w_fc2 = init_weight([1024, 10], name='wfc2')
b_fc2 = init_bias([10], name='bfc2')

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2)+b_fc2, name='yconv')


def show_img(image, label=''):
    image_reshape = image.reshape(28,28)
    img_0 = Image.fromarray(image_reshape*255)
    if label:
        print('label:%s' % np.argmax(label))
    plt.imshow(img_0)

#def prediction(sess, n=0):
#    test_img = mnist.test.images[n]
#    test_label = mnist.test.labels[n]
#    show_img(test_img, test_label)
#    y_r = sess.run(y_conv, feed_dict={x: [test_img], keep_prob: 1.0})
#    print('prediction:%s' % (np.argmax(y_r,1)[0]))

def prediction2(sess, im_arr):
#    show_img(im_arr)
    y_r = sess.run(y_conv, feed_dict={x: im_arr, keep_prob: 1.0})
    r = np.argmax(y_r,1)[0]
#    print('y_r:%s' % y_r)
#    print('prediction:%s' % r)
    return r

#img_offset(n='3.34')

def num_offs(image_reshape):
    # 数字居中
    raw_img = copy.deepcopy(image_reshape)
    try:
        cx = np.array([sum(image_reshape[:, ii]) for ii in range(28)])
        cx_e = np.argwhere(cx>0)
        
        cx_l = []
        cx_i = 0
        for x in cx_e:
            # 按序列切割
            if not cx_l:
                cx_l.append(x[0])
            else:
                if x[0] != (cx_l[cx_i] + 1):
                    cx_l.append(0)
                    cx_l.append(x[0])
                    cx_i += 2
                else:
                    cx_l.append(x[0])
                    cx_i += 1
        cx_d = []
        cx_t = []
        cx_ti = []
        for x in cx_l:
            if x != 0:
                cx_t.append(x)
            else:
                cx_d.append(cx_t)
                cx_ti.append(len(cx_t))
                cx_t = []
            if x == cx_l[-1]:
                cx_d.append(cx_t)
                cx_ti.append(len(cx_t))
                cx_t = []
        
        # 中间断层
        max_idx = cx_ti.index(max(cx_ti))
        num_idx = cx_d[max_idx]
        if (max_idx-1) >= 0 and len(cx_d[max_idx-1]) > 2:
            num_idx = cx_d[max_idx-1] + num_idx
        if (max_idx+1) <= len(cx_d)-1 and len(cx_d[max_idx+1]) > 2:
            num_idx = num_idx + cx_d[max_idx+1]
        
    #    print(num_idx)
        left_n = num_idx[0] - 0
        right_n = 27 - num_idx[-1]
        offs = int((left_n + right_n) / 2 - left_n)
    #    print(offs)
        if offs > 0:
            for i in range(27, -1, -1):
                # 填充
                if i >= offs:
                    image_reshape[:, i] = image_reshape[:, i-offs]
                else:
                    image_reshape[:, i] = 0
        elif offs < 0:
            for i in range(0, 28, 1):
                if i >= (28 + offs):
                    image_reshape[:, i] = 0
                else:
                    image_reshape[:, i] = image_reshape[:, i-offs]
        return image_reshape
    except Exception as e:
        print(e)
    return raw_img
    
#mnist.test.images[344]

#show_img(mnist.train.images[7])



saver = tf.train.Saver()

def test():
    with tf.Session() as sess: 
    #    model = tf.train.latest_checkpoint('./ckpt_cnn')
    #    saver.restore(sess, model)
        saver.restore(sess, './cnn_s3/cnn_mnist')
    #    prediction2(sess, mnist.train.images[7].reshape(1,784))
        
    #    ns = ['l1c1','l1c2','l2c1','l2c2','l3c1','l3c2','l3c3',
    #              'l3c4','l3c5','l3c6']
    #    ns = ['pls168','pls239','pls315','plw168','plw239','plw315',
    #          'plw387','plw461']
        f = 0
        s = 720
        ns = ['%s.%s' % (f, i+s) for i in range(20)]
        plt.figure(figsize=(10,7))
        for i, n in enumerate(ns):
            image = Image.open('./numdata/data/%s.jpg' % n)
    #        image = Image.open('./cv/test/%s.jpg' % n)
            image_arr = np.array(image.resize((28,28)))
            #image_arr.shape
            #image_arr / 255
            
            im_arr = image_arr.reshape((1,784)) / 255
            im_arr[im_arr<0.8] = 0
            
            # 二值化
#            im_arr = image_arr.reshape((1,784))
#            maximum = max(map(max, im_arr))
#            mininum = min(map(min, im_arr))
#            im_arr = (im_arr - mininum) * (255.0 / maximum)
#            im_arr[im_arr < 255 * 0.5] = 0
#            im_arr[im_arr != 0] = 1
            
            plt.subplot(4,5,i+1)
            image_reshape = im_arr.reshape((28,28))
            image_reshape = num_offs(image_reshape)
            img_0 = Image.fromarray(image_reshape*255)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(img_0)
            r = prediction2(sess, im_arr)
            plt.xlabel('%s(%s)' % (r, int(np.sum(im_arr))))
        plt.show()
        

def check_acc():
    data_dir = 'D:/machinelearning/codes/ml_learning/NN/numdata/data/'
    files = os.listdir(data_dir)
    
    cnt = 0
    errors = []
    tt = {i:0 for i in range(10)}
    error_d = {i:0 for i in range(10)}
    correct_d = {i:0 for i in range(10)}
    err_p = {}
    with tf.Session() as sess: 
        saver.restore(sess, './cnn_s3/cnn_mnist')
        
        
        ns = files
        for i, n in enumerate(ns):
            image = Image.open('./numdata/data/%s' % n)
            image_arr = np.array(image.resize((28,28)))
            #image_arr.shape
            #image_arr / 255
            
            im_arr = image_arr.reshape((1,784)) / 255
            im_arr[im_arr<0.8] = 0
            
            image_reshape = im_arr.reshape((28,28))
            image_reshape = num_offs(image_reshape)
            im_arr = image_reshape.reshape((1,784))
            
            # 二值化
#            im_arr = image_arr.reshape((1,784))
#            maximum = max(map(max, im_arr))
#            mininum = min(map(min, im_arr))
#            im_arr = (im_arr - mininum) * (255.0 / maximum)
#            im_arr[im_arr < 255 * 0.5] = 0
#            im_arr[im_arr != 0] = 1
            
            r = int(prediction2(sess, im_arr))
            raw_n = int(n[0])
            tt[raw_n] += 1
            if r == raw_n:
                cnt += 1
                correct_d[r] += 1
            else:
                errors.append(n)
                error_d[raw_n] += 1
        p = cnt / len(files)
        print('正确率:', p)
        print('错误数:', len(errors))
        if len(errors)<10:
            print(errors)
        print('总数分布:', tt)
        print('正确分布:', correct_d)
        print('错误分布:', error_d)
        for i in range(10):
            err_p[i] = round(error_d[i] / tt[i], 2)
        print('错误率:', err_p)
        
    err = {}
    for e in errors:
        if e[0] not in err:
            err[e[0]] = 0
        err[e[0]] += 1
        
#check_acc()
test()


def binarization(img):
    """二值化"""
    maximum = max(map(max, img))
    mininum = min(map(min, img))
    th_img = (img - mininum) * (255.0 / maximum)
    th_img = th_img.astype(np.uint8)
    th_img[th_img < 255 * 0.5] = 0
    th_img[th_img != 0] = 255
    return th_img

def img_offset(n='6.112'):
    image = Image.open('./numdata/data/%s.jpg' % n)
    #        image = Image.open('./cv/test/%s.jpg' % n)
    image_arr = np.array(image.resize((28,28)))
    #image_arr.shape
    #image_arr / 255
    
    im_arr = image_arr.reshape((1,784)) / 255
    im_arr[im_arr<0.8] = 0
    
    plt.subplot(4,5,1)
    image_reshape = im_arr.reshape(28,28)
    
    image_reshape = num_offs(image_reshape)
            
    img_0 = Image.fromarray(image_reshape*255)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(img_0)
    plt.show()
