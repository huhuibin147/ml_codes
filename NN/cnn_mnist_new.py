# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
from PIL import Image

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
y_ = tf.placeholder(tf.float32, [None, 10])


cross_entropy = -tf.reduce_mean(y_*tf.log(y_conv))
optimizer = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)


correct_prediction = tf.equal(tf.arg_max(y_conv,1), tf.arg_max(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))




# 原数据读取
print('训练数据读取...')
data_dir = 'D:/machinelearning/codes/ml_learning/NN/numdata/data1/'
files = os.listdir(data_dir)
image = []
label = []
img_name = []
for f in files:
    im = Image.open('%s%s' % (data_dir, f))
    image_arr = np.array(im.resize((28,28)))
    im_arr = image_arr.reshape((1,784)) / 255
    im_arr[im_arr<0.8] = 0
    image.append(im_arr.tolist()[0])
    label.append(f[0])
    img_name.append(f)
print('训练数据读取完毕...')


# 数据集拆分
print('拆分开始...')
train_img = []
train_label = []
test_img = []
test_label = []
#test_index = np.random.randint(1,100,size=10)
test_idxs = []
#for i in range(10):
#    for ti in test_index:
#        idx = img_name.index('%s.%s.jpg' % (i, ti))
#        test_idxs.append(idx)
for i in range(len(image)):
    if i in test_idxs:
        test_img.append(image[i])
        test_label.append(label[i])
    else:
        train_img.append(image[i])
        train_label.append(label[i])

train_images = np.array(train_img)
test_images = np.array(test_img)

train_ls = []
test_ls = []
for i in train_label:
    l = list(np.zeros(10))
    l[int(i)] = 1.0
    train_ls.append(l)
for i in test_label:
    l = list(np.zeros(10))
    l[int(i)] = 1.0
    test_ls.append(l)
    
train_labels = np.array(train_ls)
test_labels = np.array(test_ls)

# 乱序处理
per = np.random.permutation(train_images.shape[0])
train_images_s = train_images[per, :]
train_labels_s = train_labels[per, :]
print('拆分结束...')


epoches = 50
batch_size = 1024
n_batch = len(train_img) // batch_size

init = tf.global_variables_initializer()
saver = tf.train.Saver()


# 提前加载
print('提前加载开始...')
im_arrs = []
data_dir = 'D:/machinelearning/codes/ml_learning/NN/numdata/data/'
files = os.listdir(data_dir)
for i, n in enumerate(files):
    image = Image.open('./numdata/data/%s' % n)
    image_arr = np.array(image.resize((28,28)))
    im_arr = image_arr.reshape((1,784)) / 255
    im_arr[im_arr<0.8] = 0
    im_arrs.append(im_arr)
print('加载完毕...')

def check_acc(sess):
    cnt = 0
    errors = []
    ns = files
    for i, n in enumerate(ns):
        im_arr = im_arrs[i]
        
        r = prediction2(sess, im_arr)
        if int(r) == int(n[0]):
            cnt += 1
        else:
            errors.append(n)
    p = cnt / len(files)
    print('正确率:', p)
    print('错误数:', len(errors))
    if len(errors)<10:
        print(errors)
            
    err = {}
    for e in errors:
        if e[0] not in err:
            err[e[0]] = 0
        err[e[0]] += 1
    return p
        
def prediction2(sess, im_arr):
    y_r = sess.run(y_conv, feed_dict={x: im_arr, keep_prob: 1.0})
    r = np.argmax(y_r,1)[0]
    return r


with tf.Session() as sess:
    sess.run(init)
    print('init suc...')
    xx = 0
    last1 = 0
    last2 = 0
    for i in range(epoches):
        per = np.random.permutation(train_images.shape[0])
        train_images_s = train_images_s[per, :]
        train_labels_s = train_labels_s[per, :]
        for batch in range(n_batch):
            print('train batch:', batch)
            st, end = batch * batch_size, batch * batch_size + batch_size
            batch_xs, batch_ys = train_images_s[st:end, ], train_labels_s[st:end, ]
            optimizer.run(feed_dict={x:batch_xs,y_:batch_ys,keep_prob:0.5})
        if i % 1 == 0:
            train_accuacy = accuracy.eval(feed_dict={x:train_images_s[:5000],
                                          y_:train_labels_s[:5000],keep_prob:1.0})
            print("epoch:%s,acc:%s"%(i, train_accuacy))
            acc = check_acc(sess)
#            if acc > 0.99:
#                break
            if acc < last1 and acc < last2:
                break
            last2 = last1
            last1 = acc
            if train_accuacy >= 1 and acc > 0.99:
                xx += 1
            if xx >= 5:
                break
    saver.save(sess, './cnn_s3/cnn_mnist')

#    print("test accuracy %g"%(accuracy.eval(feed_dict={x: test_images, 
#                                                       y_: test_labels, keep_prob: 1.0})))
