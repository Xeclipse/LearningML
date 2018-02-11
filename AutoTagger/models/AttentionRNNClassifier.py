# coding:utf-8

import tensorflow as tf
from tensorflow.contrib.rnn import MultiRNNCell, BasicRNNCell, LSTMCell
import NeuralNetworkUtils.layers.layers as layer
import numpy as np
from NeuralNetworkUtils.utils.utils import oneHotLabels, split2Batches
from tensorFlowTest.libs.utils import weight_variable
from utils import textProcess as tp


def genToyData():
    X = np.random.random_integers(1, 10, [20, 10])
    Y = np.random.random_integers(0, 1, 20)
    lengths = np.random.random_integers(5, 10, 20)
    return X, Y, lengths


class AlbumTextClassifier:
    def __init__(self):
        self.displayStep = 1
        self.numUnits = 21
        self.vocabDim = 12
        self.vecDim = 7
        self.numLabels = 2
        self.net = None
        self.modelPath = "../Record/modelRecord/testModel.rnn"
        self.tensorBoardPath = "../Record/TensorBoard/TestTrial"
        self.graphName = "attention_rnn_cls"
        self.tensorsByName = {
            "tag": self.graphName + '/inputsTag:0',
            "title": self.graphName + '/inputsTitle:0',
            "intro": self.graphName + '/inputsIntro:0',
            'y': self.graphName + '/outputs:0',
            'pred': self.graphName + '/attention_layer/dense/hidden:0',
            'acc': self.graphName + '/attention_layer/measure/accuracy:0',
            'loss': self.graphName + '/attention_layer/loss/square_loss:0'
        }

    def attention_layer(self, value, hiddenUnits, numLabels, name):
        # todo:finish this network and build a class
        with tf.name_scope("attention_layer"):
            # y = tf.placeholder(dtype=tf.float32, shape=[None, self.numLabels], name="outputs"
            with tf.variable_scope("rnn_attention_" + name, reuse=False):
                with tf.variable_scope("rnn_encoder_variable", reuse=False):
                    cell = LSTMCell(num_units=hiddenUnits, initializer=tf.truncated_normal_initializer)
                    cell = tf.nn.rnn_cell.DropoutWrapper(cell, 0.8)
                    outputs, lastStates = layer.dynamicRnnLayer(value, cell)
                with tf.variable_scope("rnn_attention_variable", reuse=False):
                    attentionCell = LSTMCell(num_units=1, initializer=tf.truncated_normal_initializer)
                    attentionCell = tf.nn.rnn_cell.DropoutWrapper(attentionCell, 0.8)
                    attentions, lastAttentionState = layer.dynamicRnnLayer(outputs, attentionCell)
            attentionEmbedding = tf.matmul(attentions, outputs, transpose_a=True)
            squeeze = tf.squeeze(attentionEmbedding)
            predict = layer.dense_layer(squeeze, weightShape=[hiddenUnits, numLabels],
                                        outputShape=[numLabels], name=name)
            return predict

    def createModel(self):
        with tf.name_scope(self.graphName):
            x1 = tf.placeholder(dtype=tf.int32, shape=[None, None], name="inputsTag")
            x2 = tf.placeholder(dtype=tf.int32, shape=[None, None], name="inputsTitle")
            x3 = tf.placeholder(dtype=tf.int32, shape=[None, None], name="inputsIntro")
            y = tf.placeholder(dtype=tf.float32, shape=[None, self.numLabels], name="outputs")
            embeddingTable = weight_variable([self.vocabDim, self.vecDim])
            tf.summary.histogram(name='embedding', values=embeddingTable)

            embeddingTag = tf.nn.embedding_lookup(embeddingTable, x1)
            embeddingTitle = tf.nn.embedding_lookup(embeddingTable, x2)
            embeddingIntro = tf.nn.embedding_lookup(embeddingTable, x3)

            introPredict = self.attention_layer(value=embeddingTag, hiddenUnits=50, numLabels=self.numLabels,
                                                name="introPredict")
            titlePredict = self.attention_layer(value=embeddingTitle, hiddenUnits=25, numLabels=self.numLabels,
                                                name="titlePredict")
            tagsPredict = self.attention_layer(value=embeddingIntro, hiddenUnits=25, numLabels=self.numLabels,
                                               name="tagsPredict")
            concat = tf.concat([introPredict, titlePredict, tagsPredict], 1)
            mixedPredict = layer.dense_layer(concat, weightShape=[self.numLabels * 3, self.numLabels],
                                             outputShape=[self.numLabels], name="mixedPredict")
            lossIntro = layer.square_loss(y, introPredict)
            lossTitle = layer.square_loss(y, titlePredict)
            lossTags = layer.square_loss(y, titlePredict)
            lossMixed = layer.square_loss(mixedPredict, y)
            lossConcat = tf.stack([lossIntro, lossTitle, lossTags, lossMixed], axis=0)
            lossWeight = tf.constant(value=[1, 1, 1, 3], dtype=tf.float32)
            totalLoss = tf.reduce_sum(tf.multiply(lossConcat, lossWeight))
            optimizer = layer.optimize_op(totalLoss, rate=0.001)
            return {'optimizer': optimizer, 'totalLoss': totalLoss, 'lossTags': lossTags, 'lossTitle': lossTitle,
                    'lossIntro': lossIntro}

    # X should be an albums features
    # X = [
    # [ tags] , [titles], [intros]
    # ]
    def train(self, X, Y, batch_size=1000, epoch=200):
        if self.net is None:
            self.net = self.createModel()
        batchTags, batchY, batchNum = split2Batches(batch_size, X[0], Y)
        tp.batchPadding(batchTags)
        batchTitles, batchY, batchNum = split2Batches(batch_size, X[1], Y)
        tp.batchPadding(batchTitles)
        batchIntros, batchY, batchNum = split2Batches(batch_size, X[2], Y)
        tp.batchPadding(batchIntros)
        with tf.Session() as sess:
            summary = tf.summary.FileWriter(self.tensorBoardPath)
            summary.add_graph(sess.graph, 1)
            sess.run(tf.global_variables_initializer())
            obop = tf.summary.merge_all()
            tag = tf.get_default_graph().get_tensor_by_name(self.tensorsByName['tag'])
            title = tf.get_default_graph().get_tensor_by_name(self.tensorsByName['title'])
            intro = tf.get_default_graph().get_tensor_by_name(self.tensorsByName['intro'])
            y = tf.get_default_graph().get_tensor_by_name(self.tensorsByName['y'])
            sess.run(tf.global_variables_initializer())
            for i in range(epoch):
                epochSta = [0] * 5
                for k in range(batchNum):
                    res = sess.run(
                        [self.net['optimizer'], self.net['totalLoss'], self.net['lossTags'], self.net['lossTitle'],
                         self.net['lossIntro']],
                        feed_dict={tag: batchTags[k], title: batchTitles[k], intro: batchIntros[k], y: batchY[k]})
                    epochSta[1] += res[1]
                    epochSta[2] += res[2]
                    epochSta[3] += res[3]
                    epochSta[4] += res[4]
                    if i % self.displayStep == 0:
                        print '=',
                epochSta = [str(i / batchNum) for i in epochSta]
                # if i % self.displayStep == 0:
                print '\t|\t'.join(epochSta[1:])
                try:
                    ops = sess.run(obop, feed_dict={tag: batchTags[0], title: batchTitles[0], intro: batchIntros[0],
                                                    y: batchY[0]})

                    summary.add_summary(ops, i)
                except:
                    pass
                    # print epochSta[1] / batchNum, '\t|\t', epochSta[2] / batchNum
            saver = self.saveModel(sess)
            sess.close()

    # todo: finish this methods
    def saveModel(self, sess):
        saver = tf.train.Saver()
        saver.save(sess, self.modelPath)
        return saver

    def readModel(self, sess):
        saver = tf.train.import_meta_graph(self.modelPath + '.meta')
        saver.restore(sess, self.modelPath)
        return saver

    def predict(self, X):
        pass

# album = AlbumTextClassifier()
# album.createModel()
# X, Y, length = genToyData()
# Y = oneHotLabels(Y.tolist())
# print Y
# atten = AttentionClassifier()
# with tf.Session() as sess:
#     atten.train(X, Y)
#     sess.close()
