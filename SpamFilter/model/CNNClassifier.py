# coding:utf-8

import tensorflow as tf
from scipy.sparse import coo_matrix
from sklearn.utils import shuffle

from SpamFilter import model, preprocess
from tensorFlowTest.libs.utils import weight_variable, bias_variable
import numpy as np
import math
from utils import textProcess


def embedding_layer(value, vocabDim, vecDim):
    with tf.name_scope("embedding"):
        W = weight_variable([vocabDim, vecDim])
        embeddingVactor = tf.nn.embedding_lookup(params=W, ids=value)
        tf.summary.histogram(name='embedding', values=W)
        return embeddingVactor


def dense_layer(value, weightShape, outputShape):
    with tf.name_scope("dense"):
        weight = weight_variable(shape=weightShape)
        bias = bias_variable(shape=outputShape)
        hidden = tf.nn.relu(tf.matmul(value, weight) + bias)
        tf.summary.histogram(name='denseW', values=weight)
        tf.summary.histogram(name='denseBias', values=bias)
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
        tf.summary.histogram(name='convW', values=wConvLayer)
        tf.summary.histogram(name='convBias', values=bConvLayer)
        return hidden


def softmax_layer(value, inputShape, outputShape):
    with tf.name_scope("softmax"):
        wSoftmax = weight_variable(shape=inputShape)
        bSoftmax = bias_variable(shape=outputShape)
        ypred = tf.nn.softmax(tf.matmul(value, wSoftmax) + bSoftmax, name='ypred')
        tf.summary.histogram(name='softmaxW', values=wSoftmax)
        tf.summary.histogram(name='softmaxB', values=bSoftmax)
        return ypred


def dropout_layer(value):
    with tf.name_scope("dropout"):
        dropout = tf.nn.dropout(x=value, keep_prob=1.0)
        return dropout


def optimize_op(value):
    with tf.name_scope("optimizer"):
        adam = tf.train.AdagradOptimizer(learning_rate=0.01)
        return adam.minimize(value, name='adam_optimizer')


def flatten(value, flattenShape):
    with tf.name_scope("flatten"):
        return tf.reshape(value, shape=flattenShape)


def accuracy(ypred, y):
    with tf.name_scope("measure"):
        correct_prediction = tf.equal(tf.argmax(ypred, 1), tf.argmax(y, 1))
        acc = tf.reduce_mean(tf.cast(correct_prediction, 'float'), name='accuracy')
        tf.summary.scalar(name="accuracy", tensor=acc)
        return acc


def cost_loss(ypred, y):
    with tf.name_scope("loss"):
        sub = ypred - y
        pow2 = tf.multiply(sub, sub, name="pow2")
        L2 = tf.reduce_mean(input_tensor=pow2, axis=1, keep_dims=True)
        costMatrix = tf.constant([[5], [1]], dtype=tf.float32)
        costList = tf.matmul(y, costMatrix)
        loss = tf.reduce_mean(tf.matmul(L2, costList, transpose_a=True), name='cost_loss')
        tf.summary.scalar(name="cost_loss", tensor=loss)
        return loss


def cross_entropy_loss(ypred, y):
    with tf.name_scope("loss"):
        cost = -tf.reduce_mean(y * tf.log(ypred), name="cross_entropy")
        tf.summary.scalar(name="cross_entropy", tensor=cost)
        return cost


def conv_flatten_dense_layer(value, filtersize, nFilter, stride, channels, outdim):
    sendim = int(value.get_shape().as_list()[1])
    with tf.name_scope("packageLayer"):
        hidden = conv_1d_layer(value=value, filterSize=filtersize, nFilter=nFilter, stride=stride, channels=channels)
        dim = int(math.ceil(1.0 * sendim / stride)) * nFilter
        flat = flatten(value=hidden, flattenShape=[-1, dim])
        dense = dense_layer(value=flat, weightShape=[dim, outdim], outputShape=[outdim])
        return dense


def split2Batches(batchSize, X, Y):
    batchX = []
    batchY = []
    batchNum = int(math.ceil(1.0 * X.__len__() / batchSize))
    for i in range(batchNum):
        start = batchSize * i
        end = min([batchSize * (i + 1), len(X)])
        batchX.append(X[start:end])
        batchY.append(Y[start:end])
    return batchX, batchY, batchNum


def conv_net(x, y, vocabDim):
    charvecs = 2
    embedding = embedding_layer(value=x, vocabDim=vocabDim, vecDim=charvecs)
    hidden1 = conv_1d_layer(value=embedding, filterSize=4, nFilter=5, stride=3, channels=charvecs)
    dim = (50 / 3 + 1) * 5
    flat = flatten(hidden1, [-1, dim])
    dense = dense_layer(value=flat, weightShape=[dim, 10], outputShape=[10])
    ypred = softmax_layer(value=dense, inputShape=[10, 2], outputShape=[2])

    #   代价学习损失函数
    loss = cost_loss(ypred, y)

    adam = tf.train.AdagradOptimizer(learning_rate=0.01)
    optimizer = adam.minimize(loss, name='optimizer')
    acc = accuracy(ypred, y)
    return optimizer, cost_loss, acc, ypred


def conv_net2(x, y, vecdim):
    with tf.name_scope("conv2"):
        charvecs = 7
        embedding = embedding_layer(value=x, vocabDim=vecdim, vecDim=charvecs)
        hidden1 = conv_1d_layer(value=embedding, filterSize=4, nFilter=7, stride=3, channels=charvecs)
        hidden2 = conv_1d_layer(value=hidden1, filterSize=2, nFilter=3, stride=1, channels=7)
        dim = ((50 / 3 + 1) / 1) * 3
        flat = flatten(hidden2, [-1, dim])
        dense = dense_layer(value=flat, weightShape=[dim, 10], outputShape=[10])
        ypred = softmax_layer(value=dense, inputShape=[10, 2], outputShape=[2])

        loss = cost_loss(ypred, y)

        optimizer = optimize_op(loss)
        acc = accuracy(ypred, y)

        return optimizer, loss, acc, ypred


def conv_net3(x, y, vocabDim):
    with tf.name_scope("conv3"):
        charvecs = 4
        embedding = embedding_layer(value=x, vocabDim=vocabDim, vecDim=charvecs)
        hidden1 = conv_flatten_dense_layer(value=embedding, filtersize=2, nFilter=10, stride=1, channels=charvecs,
                                           outdim=20)
        hidden2 = conv_flatten_dense_layer(value=embedding, filtersize=3, nFilter=10, stride=2, channels=charvecs,
                                           outdim=20)
        hidden3 = conv_flatten_dense_layer(value=embedding, filtersize=4, nFilter=10, stride=3, channels=charvecs,
                                           outdim=20)
        cat = tf.concat([hidden1, hidden2, hidden3], axis=1, name="concat")
        catDim = cat.get_shape().as_list()[1]
        ypred = softmax_layer(value=cat, inputShape=[catDim, 2], outputShape=[2])

        loss = cost_loss(ypred, y)

        optimizer = optimize_op(loss)
        acc = accuracy(ypred, y)

        return optimizer, loss, acc, ypred


def runStartGraph(param, X_data, Y_data, vocabDim, batchSize=10000):
    x = tf.placeholder(dtype=tf.int32, shape=[None, param["senDim"]], name='x')
    y = tf.placeholder(dtype=tf.float32, shape=[None, 2], name='y')
    batchX, batchY, batchNum = split2Batches(batchSize, X_data, Y_data)
    with tf.Session() as sess:
        optimizer, loss, acc, ypred = conv_net3(x=x, y=y, vocabDim=vocabDim)
        writer = tf.summary.FileWriter(param["writerPath"])
        writer.add_graph(graph=sess.graph, global_step=displayStep)
        obops = tf.summary.merge_all()
        sess.run(tf.global_variables_initializer())
        for i in range(2):
            for batch in range(batchNum):
                sess.run(optimizer, feed_dict={x: batchX[batch], y: batchY[batch]})
                if(batch%(batchNum/10))==0:
                    print '=',
            print 'Iter ',i,' finished'
            if (i % displayStep == 0):
                a = sess.run(acc, feed_dict={x: X_data, y: Y_data})
                l = sess.run(loss, feed_dict={x: X_data, y: Y_data})
                ops = sess.run(obops, feed_dict={x: X_data, y: Y_data})
                writer.add_summary(ops, i)
                print "acc:",
                print a,
                print "loss:",
                print l
        saver = tf.train.Saver()
        saver.save(sess, paramStart["savePath"])
        writer.close()
        sess.close()


def runFromSaver(param, X_data, Y_data, batchSize):
    batchX, batchY, batchNum = split2Batches(batchSize, X_data, Y_data)
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(param["loadPath"])
        saver.restore(sess, param["savePath"])
        writer = tf.summary.FileWriter(param["writerPath"])
        writer.add_meta_graph(saver.export_meta_graph())
        tensors = {}
        for k, v in param["getTensorNames"].items():
            tensors[k] = tf.get_default_graph().get_tensor_by_name(v)
        ops = {}
        for k, v in param["getOperationNames"].items():
            ops[k] = tf.get_default_graph().get_operation_by_name(v)
        obops = tf.summary.merge_all()
        for i in range(param["iteration"]):
            for batch in range(batchNum):
                sess.run(ops["optimizer"], feed_dict={tensors["x"]: batchX[batch], tensors["y"]: batchY[batch]})
                if (batch % (batchNum / 10)) == 0:
                    print '=',
            print 'Iter ', i, ' finished'
            if (i % displayStep == 0):
                a = sess.run(tensors["acc"], feed_dict={tensors["x"]: X_data, tensors["y"]: Y_data})
                l = sess.run(tensors["loss"], feed_dict={tensors["x"]: X_data, tensors["y"]: Y_data})
                ob = sess.run(obops, feed_dict={tensors["x"]: X_data, tensors["y"]: Y_data})
                writer.add_summary(ob, param["startIteration"] + i)
                print "acc:",
                print a,
                print "loss:",
                print l
        saver = tf.train.Saver()
        saver.save(sess, param["savePath"])
        sess.close()


def predict(param={}, X_data=None, Y_data=None):
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(param["loadPath"])
        saver.restore(sess, param["savePath"])
        writer = tf.summary.FileWriter(param["writerPath"])
        writer.add_meta_graph(saver.export_meta_graph())

        tensors = {}
        for k, v in param["getTensorNames"].items():
            tensors[k] = tf.get_default_graph().get_tensor_by_name(v)
        ops = {}
        for k, v in param["getOperationNames"].items():
            ops[k] = tf.get_default_graph().get_operation_by_name(v)

        ypred = sess.run(tensors["ypred"], feed_dict={tensors["x"]: X_data, tensors["y"]: Y_data})
        sess.close()
        return ypred



displayStep = 5

saveName = "cost_conv3_large"
graphName = "conv3"
dir = '/Users/nali/PycharmProjects/LearningML'

paramStart = {
    "writerPath": dir + '/tensorFlowTest/tensorBoard/' + saveName,
    "savePath": dir + '/SpamFilter/save/' + saveName + '/spam.cnn',
    "senDim": 40
}

paramFromSaver = {
    "writerPath": dir + '/tensorFlowTest/tensorBoard/' + saveName,
    "loadPath": dir + '/SpamFilter/save/' + saveName + '/spam.cnn.meta',
    "savePath": dir + '/SpamFilter/save/' + saveName + '/spam.cnn',
    "getTensorNames": {
        "acc": graphName + "/measure/accuracy:0",
        "loss": graphName + "/loss/cost_loss:0",
        "x": "x:0",
        "y": "y:0",
        "ypred": graphName + "/softmax/ypred:0"
    },
    "getOperationNames": {
        "optimizer": graphName + "/optimizer/adam_optimizer",
    },
    "iteration": 48,
    "startIteration": 2,
    "senDim": 40
}

nospam = "/Users/nali/PycharmProjects/LearningML/SpamFilter/Data/search-filter-no-spam.log"
spam = "/Users/nali/PycharmProjects/LearningML/SpamFilter/Data/search-filter-spam.log"
test = "/Users/nali/PycharmProjects/LearningML/SpamFilter/Data/search-filter-data.log"
dic ={}
dic = textProcess.loadDict('/Users/nali/PycharmProjects/LearningML/SpamFilter/Data/dict')
rev = textProcess.reverseDic(dic)
X, Y, dic = preprocess.generateTestData(nospam, dic, 40)
X2, Y2, dic = preprocess.generateTestData(spam, dic, 40)
X3, Y3, dic = preprocess.generateTestData(test, dic, 40)
# textProcess.saveDict(dic, '/Users/nali/PycharmProjects/LearningML/SpamFilter/Data/dict')
X.extend(X2)
Y.extend(Y2)
X.extend(X3)
Y.extend(Y3)
X, Y = preprocess.shuffleData(X, Y)
print np.sum(Y, 0)

runFromSaver(param=paramFromSaver, X_data=X, Y_data=Y, batchSize=10000)
# runStartGraph(param=paramStart, X_data=X, Y_data=Y, vocabDim= dic.__len__()+2, batchSize= 10000)
# ypred = predict(param=paramFromSaver,X_data= X, Y_data=Y)
# x = X
# y = Y
# for i, v in enumerate(ypred):
#     if np.argmax(v) != np.argmax(y[i]):
#         print y[i], '\t', textProcess.id2String(x[i], rev)