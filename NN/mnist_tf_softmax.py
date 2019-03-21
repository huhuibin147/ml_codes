# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

mnist_data = input_data.read_data_sets("./MNIST_DATA/", one_hot=True)
#dir(mnist_data.train)

mnist_data.train.images
mnist_data.train.labels

images_0 = mnist_data.train.images[0]
labels_0 = mnist_data.train.labels[0]


def show_img(image, label):
    image_reshape = image.reshape(28,28)
    img_0 = Image.fromarray(image_reshape*255)
    print('label:%s' % np.argmax(label))
    plt.imshow(img_0)
    
show_img(mnist_data.train.images[7], mnist_data.train.labels[7])


W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
x = tf.placeholder(tf.float32, [None, 784])

y = tf.nn.softmax(tf.matmul(x,W)+b)
y_ = tf.placeholder(tf.float32, [None, 10])

cross_entropy = -tf.reduce_mean(y_*tf.log(y))
optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

correct_prediction = tf.equal(tf.arg_max(y,1), tf.arg_max(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.global_variables_initializer()

saver = tf.train.Saver(max_to_keep=1)

epoches = 400
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
    saver.save(sess, './ckpt/mnist.ckpt')
            

isess = tf.InteractiveSession()

model = tf.train.latest_checkpoint('./ckpt')
saver.restore(isess, model)

def prediction(n=0):
    test_img = mnist_data.test.images[n]
    test_label = mnist_data.test.labels[n]
    show_img(test_img, test_label)
    y_r = isess.run(y, feed_dict={x: [test_img]})
    print('prediction:%s' % (np.argmax(y_r,1)[0]))

prediction(133)




#print(isess.run(tf.arg_max([[1,2,3]],0)))
#print(isess.run(tf.arg_max([[1,2,3],[2,2,4]],0)))
#print(isess.run(tf.arg_max([[1,2,3],[2,2,4]],1)))
