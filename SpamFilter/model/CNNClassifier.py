# coding:utf-8

import tensorflow as tf

from SpamFilter import model, preprocess
from tensorFlowTest.libs.utils import weight_variable, bias_variable
import numpy as np
import utils


def embedding_layer(value, vocabDim, vecDim):
    with tf.name_scope("embedding"):
        W = weight_variable([vocabDim, vecDim])
        embeddingVactor = tf.nn.embedding_lookup(params=W, ids=value)
        return embeddingVactor


def dense_layer(value, weightShape, outputShape):
    with tf.name_scope("dense"):
        weight = weight_variable(shape=weightShape)
        bias = bias_variable(shape=outputShape)
        hidden = tf.nn.relu(tf.matmul(value, weight) + bias)
        return hidden


def conv_1d_layer(value, filterSize, nFilter, stride, channels):
    with tf.name_scope("conv1d"):
        stride = stride
        filter_size = filterSize
        n_filter = nFilter
        wConvLayer = weight_variable([filter_size, channels, n_filter])
        bConvLayer = bias_variable([n_filter])
        hidden = tf.nn.relu(
            tf.nn.conv1d(
                value=value,
                filters=wConvLayer,
                stride=stride,
                padding='SAME') +
            bConvLayer)
        return hidden


def softmax_layer(value, inputShape, outputShape):
    with tf.name_scope("softmax"):
        wSoftmax = weight_variable(shape=inputShape)
        bSoftmax = bias_variable(shape=outputShape)
        ypred = tf.nn.softmax(tf.matmul(value, wSoftmax) + bSoftmax, name='ypred')
        return ypred


def dropout_layer(value):
    with tf.name_scope("dropout"):
        dropout = tf.nn.dropout(x=value, keep_prob=1.0)
        return dropout

def optimize_op(value):
    with tf.name_scope("optimizer"):
        adam = tf.train.AdagradOptimizer(learning_rate=0.01)
        return adam.minimize(value, name='adam_optimizer')

def conv_net2(x,y,vecdim):
    with tf.name_scope("conv2"):
        charvecs = 7
        embedding = embedding_layer(value=x, vocabDim=vecdim, vecDim=charvecs)
        hidden1 = conv_1d_layer(value=embedding, filterSize=4, nFilter=7, stride=3, channels=charvecs)
        hidden2 = conv_1d_layer(value=hidden1, filterSize=2, nFilter=3, stride=1, channels=7)
        dim = ((50 / 3 + 1) / 1) * 3
        flat =flatten(hidden2, [-1, dim])
        dense = dense_layer(value=flat, weightShape=[dim, 10], outputShape=[10])
        ypred = softmax_layer(value=dense, inputShape=[10, 2], outputShape=[2])

        loss = cost_loss(ypred, y)

        optimizer = optimize_op(loss)
        acc = accuracy(ypred, y)
        return optimizer, loss, acc, ypred

def flatten(value, flattenShape):
    with tf.name_scope("flatten"):
        return tf.reshape(value, shape=flattenShape)


def cost_loss(ypred, y):
    with tf.name_scope("loss"):
        cost_matrix = tf.constant(value=[[0.1], [40.0]], dtype=tf.float32)
        diag = tf.diag_part(tf.matmul(y - ypred, y - ypred, transpose_a=True))
        diag = tf.reshape(tensor=diag, shape=[1, 2])
        loss = tf.squeeze(tf.matmul(diag, cost_matrix), name='cost_loss')
        return loss


def cross_entropy_loss(ypred, y):
    with tf.name_scope("loss"):
        return -tf.reduce_mean(y * tf.log(ypred), name="cross_entropy")
        # L2 = tf.trace(tf.matmul(y - ypred, y - ypred, transpose_a=True))

def conv_net(x, y, vecdim):
    charvecs = 2
    embedding = embedding_layer(value=x, vocabDim=vecdim, vecDim=charvecs)
    hidden1 = conv_1d_layer(value=embedding, filterSize=4, nFilter=5, stride=3, channels=charvecs)
    # hidden2 = conv_1d_layer(value=hidden1, filterSize=3, nFilter=5, stride=5, channels=10)
    dim = (50 / 3 + 1) * 5
    flat = flatten(hidden1, [-1, dim])
    dense = dense_layer(value=flat, weightShape=[dim, 10], outputShape=[10])
    ypred = softmax_layer(value=dense, inputShape=[10, 2], outputShape=[2])

    #   代价学习损失函数
    loss = cost_loss(ypred,y)

    adam = tf.train.AdagradOptimizer(learning_rate=0.01)
    optimizer = adam.minimize(loss, name='optimizer')
    acc = accuracy(ypred, y)
    return optimizer, cost_loss, acc, ypred


def accuracy(ypred, y):
    with tf.name_scope("measure"):
        correct_prediction = tf.equal(tf.argmax(ypred, 1), tf.argmax(y, 1))
        acc = tf.reduce_mean(tf.cast(correct_prediction, 'float'), name='accuracy')
        print acc.name
        return acc


displayStep = 2

X_data, Y_data, dic = preprocess.generateTrainData()
# vecdim = dic.__len__()
#
# x = tf.placeholder(dtype=tf.int32, shape=[None, 50], name='x')
# y = tf.placeholder(dtype=tf.float32, shape=[None, 2], name='y')

# with tf.Session() as sess:
#     optimizer, loss, acc, ypred = conv_net2(x=x, y=y, vecdim=vecdim)
#     writer = tf.summary.FileWriter('/Users/nali/PycharmProjects/LearningML/tensorFlowTest/tensorBoard/conv2')
#     writer.add_graph(graph=sess.graph)
#
#     sess.run(tf.global_variables_initializer())
#     for i in range(2):
#         sess.run(optimizer, feed_dict={x: X_data, y: Y_data})
#         if (i % displayStep == 0):
#             a = sess.run(acc, feed_dict={x: X_data, y: Y_data})
#             l = sess.run(loss, feed_dict={x: X_data, y: Y_data})
#             print "acc:",
#             print a,
#             print "loss:",
#             print l
#     saver = tf.train.Saver()
#     saver.save(sess, '/Users/nali/PycharmProjects/LearningML/SpamFilter/save/conv2/spam.cnn')
#     sess.close()

saver = tf.train.import_meta_graph('/Users/nali/PycharmProjects/LearningML/SpamFilter/save/conv2/spam.cnn.meta')
sess = tf.Session()
saver.restore(sess, '/Users/nali/PycharmProjects/LearningML/SpamFilter/save/conv2/spam.cnn')
X_data, Y_data, dic = preprocess.generateTrainData()
y=tf.get_default_graph().get_tensor_by_name("y:0")
x=tf.get_default_graph().get_tensor_by_name("x:0")
acc = tf.get_default_graph().get_tensor_by_name("conv2/measure/accuracy:0")
loss = tf.get_default_graph().get_tensor_by_name("conv2/loss/cost_loss:0")
optimizer = tf.get_default_graph().get_operation_by_name("conv2/optimizer/adam_optimizer")

ypred = tf.get_default_graph().get_tensor_by_name("conv2/softmax/ypred:0")
for i in range(4):
    sess.run(optimizer, feed_dict={x: X_data, y: Y_data})

    if (i % displayStep == 0):
        a = sess.run(acc, feed_dict={x: X_data, y: Y_data})
        l = sess.run(loss, feed_dict={x: X_data, y: Y_data})
        print "acc:",
        print a,
        print "loss:",
        print l
saver = tf.train.Saver()
saver.save(sess,'/Users/nali/PycharmProjects/LearningML/SpamFilter/save/conv2/spam.cnn')
sess.close()
