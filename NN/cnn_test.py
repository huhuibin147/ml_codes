# -*- coding: utf-8 -*-

import tensorflow as tf
#from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import os
os.chdir("D:/machinelearning/codes/ml_learning/NN/")
os.getcwd()

#MNIST_DATA_DIR = "D:/machinelearning/MNIST/MNIST_data/"

#mnist = input_data.read_data_sets(MNIST_DATA_DIR, one_hot=True)


#sess = tf.InteractiveSession()



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


x = tf.placeholder(tf.float32, [None, 784])
x_image = tf.reshape(x, [-1,28,28,1])

w_conv1 = init_weight([7,7,1,32], name='w1')
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
#y_ = tf.placeholder(tf.float32, [None, 10])


#cross_entropy = -tf.reduce_mean(y_*tf.log(y_conv))
#optimizer = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
#
#
#correct_prediction = tf.equal(tf.arg_max(y_conv,1), tf.arg_max(y_,1))
#accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#sess.run(tf.global_variables_initializer())

#epoches = 15
#batch_size = 512
#n_batch = mnist.train.num_examples // batch_size
#
#
#init = tf.global_variables_initializer()
#saver = tf.train.Saver()

#with tf.Session() as sess:
#    sess.run(init)
#    print('init suc...')
#    for i in range(epoches):
#        for batch in range(n_batch):
#            print('train batch:', batch)
#            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
#            optimizer.run(feed_dict={x:batch_xs,y_:batch_ys,keep_prob:0.5})
#        if i % 1 == 0:
#            train_accuacy = accuracy.eval(feed_dict={x:mnist.test.images,
#                                          y_:mnist.test.labels,keep_prob:1.0})
#            print("epoch:%s,acc:%s"%(i, train_accuacy))
#    saver.save(sess, './ckpt_cnn/cnn_mnist.ckpt')

#    print("test accuracy %g"%(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})))


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
    print(im_arr)
    y_r = sess.run(y_conv, feed_dict={x: im_arr, keep_prob: 1.0})
    r = np.argmax(y_r,1)[0]
    print('y_r:%s' % y_r)
    print('prediction:%s' % r)
    return r

    
    
#mnist.test.images[344]

#show_img(mnist.train.images[7])

#tf.reset_default_graph()
saver = tf.train.Saver()

with tf.Session() as sess: 
#    model = tf.train.latest_checkpoint('./ckpt_cnn')
#    saver.restore(sess, model)
    tf.global_variables_initializer()
    saver.restore(sess, './cnn/cnn_mnist')
#    prediction2(sess, mnist.train.images[7].reshape(1,784))
    

    ns = ['pls168','pls239','pls315','plw168','plw239','plw315',
          'plw387','plw461']
    x = 0
    st = 1
#    ns = ['%s.%s' % (x, st+i) for i in range(8)]
#    ns = ['0.1','0.2','0.3','0.4','0.5','0.6']
    
    plt.figure(figsize=(10,5))
    for i, n in enumerate(ns):
#        image = Image.open('./numdata/data/%s.jpg' % n)
        image = Image.open('./cv/test/%s.jpg' % n)
        image_arr = np.array(image.resize((28,28)))
        #image_arr.shape
        #image_arr / 255
        im_arr = image_arr.reshape((1,784)) / 255
        im_arr[im_arr<0.5] = 0
        
        plt.subplot(2,5,i+1)
        image_reshape = im_arr.reshape(28,28)
        img_0 = Image.fromarray(image_reshape*255)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(img_0)
        r = prediction2(sess, im_arr)
        plt.xlabel('%s(%s)' % (r, int(np.sum(im_arr))))
    plt.show()
        

