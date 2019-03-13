# -*- coding: utf-8 -*-
import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


mnist_data = input_data.read_data_sets("./MNIST_DATA/", one_hot=True)



x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])


W1 = tf.Variable(tf.random_normal([784, 30]))
b1 = tf.Variable(tf.zeros([30]))
a1 = tf.nn.tanh(tf.matmul(x,W1)+b1)

W2 = tf.Variable(tf.random_normal([30, 10]))
b2 = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(a1, W2)+b2)

cross_entropy = tf.reduce_mean(tf.square(y_-y))
optimizer = tf.train.AdamOptimizer(0.01).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.global_variables_initializer()
saver = tf.train.Saver(max_to_keep=1)


epoches = 300
batch_size = 100
n_batch = mnist_data.train.num_examples // batch_size

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(epoches):
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist_data.train.next_batch(batch_size)
            sess.run(optimizer, feed_dict={x: batch_xs, y_: batch_ys})
        if epoch % 1 == 0:
            print('epoch:%s,acc:%s' % (epoch, sess.run(accuracy, 
                feed_dict={x: mnist_data.test.images, y_: mnist_data.test.labels})))
    saver.save(sess, './ckpt_nn/mnist.ckpt')



