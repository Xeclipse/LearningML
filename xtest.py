# coding:utf-8


import NeuralNetworkUtils.layers.layers as la
import tensorflow as tf


# x = tf.placeholder(tf.float32,shape=[None, 2])
# y = tf.placeholder(tf.float32, shape=[None,2])
# ret = la.softmax_layer(x,inputShape=[2,10],outputShape=[10])
#
# X = [[1.0,1.0], [0.0,0.0]]
# Y = [[1.0,1.0], [0.0,1.0]]
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     print sess.run(ret, feed_dict={x:X, y:Y})

import numpy as np
print 1/15.0