# coding:utf-8

import tensorflow as tf
from tensorflow.contrib.rnn import MultiRNNCell, BasicRNNCell, LSTMCell
import NeuralNetworkUtils.layers.layers as layer
import numpy as np
from NeuralNetworkUtils.utils.utils import oneHotLabels, split2Batches
from tensorFlowTest.libs.utils import weight_variable
from utils import textProcess as tp


class HierarchicalAlbumTextClassifier:
    def __init__(self):
        self.displayStep = 1
        self.numUnits = 21
        self.vocabDim = None
        self.vecDim = 10
        self.numLabelsLevel1 = 2
        self.numLabelsLevel2 = 2
        self.net = None
        self.modelPath = "../Record/modelRecord/testModel.rnn"
        self.tensorBoardPath = "../Record/TensorBoard/TestTrial"
        self.graphName = "attention_rnn_cls"
        self.tensorsByName = {
            "tag": self.graphName + '/inputsTag:0',
            "title": self.graphName + '/inputsTitle:0',
            "intro": self.graphName + '/inputsIntro:0',

            'labelLevel1': self.graphName + '/labelLevel1:0',
            'labelLevel2': self.graphName + '/labelLevel2:0',
            'l2pred': self.graphName + '/dense_4/L2pred:0',
            'l1pred': self.graphName + '/softmax/L1pred:0',
            'lossL1': self.graphName + '/loss/cross_entropy:0',
            'lossL2': self.graphName + '/loss_1/square_loss:0',

        }
        self.opByName = {
            'optiL1': self.graphName + '/optimizer/adam_optimizer',
            'optiL2': self.graphName + '/optimizer_1/adam_optimizer',
        }

    def createModel(self):
        hiddenNums = 100
        with tf.name_scope(self.graphName):
            x1 = tf.placeholder(dtype=tf.int32, shape=[None, None], name="inputsTag")
            x2 = tf.placeholder(dtype=tf.int32, shape=[None, None], name="inputsTitle")
            x3 = tf.placeholder(dtype=tf.int32, shape=[None, None], name="inputsIntro")
            y1 = tf.placeholder(dtype=tf.float32, shape=[None, self.numLabelsLevel1], name="labelLevel1")
            y2 = tf.placeholder(dtype=tf.float32, shape=[None, self.numLabelsLevel2], name="labelLevel2")
            intialembedding = tf.random_normal(shape=[self.vocabDim, self.vecDim], mean=0.0, stddev=0.1)
            embeddingTable = tf.Variable(intialembedding, name='embeddingTable')
            tf.summary.histogram(name='embedding', values=embeddingTable)

            # process titles
            embeddingTitle = tf.nn.embedding_lookup(embeddingTable, x2, max_norm=1.0)
            titleEmbeddings = layer.conv_1d_layer(embeddingTitle, filterSize=2, nFilter=10, stride=1,
                                                  channels=self.vecDim)

            titleEmbedding = tf.reduce_mean(
                layer.attention_layer(value=titleEmbeddings, hiddenUnits=hiddenNums, name="titleEmbedding"), axis=1)

            # process tags
            embeddingTag = tf.nn.embedding_lookup(embeddingTable, x1, max_norm=1.0)
            tagEmbeddings2 = layer.conv_1d_layer(embeddingTag, filterSize=2, nFilter=10, stride=1, channels=self.vecDim)
            tagEmbedding2 = tf.reduce_mean(
                layer.attention_layer(value=tagEmbeddings2, hiddenUnits=hiddenNums, name="tagsEmbedding2"), axis=1)
            tagEmbeddings3 = layer.conv_1d_layer(embeddingTag, filterSize=3, nFilter=10, stride=1, channels=self.vecDim)
            tagEmbedding3 = tf.reduce_mean(
                layer.attention_layer(value=tagEmbeddings3, hiddenUnits=hiddenNums, name="tagsEmbedding3"), axis=1)
            tagEmbeddings4 = layer.conv_1d_layer(embeddingTag, filterSize=4, nFilter=10, stride=2, channels=self.vecDim)
            tagEmbedding4 = tf.reduce_mean(
                layer.attention_layer(value=tagEmbeddings4, hiddenUnits=hiddenNums, name="tagsEmbedding4"), axis=1)
            tagEmbedding = tf.concat([tagEmbedding2, tagEmbedding3, tagEmbedding4], axis=1)

            # process intross
            embeddingIntro = tf.nn.embedding_lookup(embeddingTable, x3, max_norm=1.0)
            introEmbeddings3 = layer.conv_1d_layer(value=embeddingIntro, filterSize=3, nFilter=20, stride=2,
                                                   channels=self.vecDim)
            introEmbedding3 = tf.reduce_mean(
                layer.attention_layer(introEmbeddings3, hiddenUnits=hiddenNums, name='introEmbedding3'), axis=1)

            introEmbeddings5 = layer.conv_1d_layer(value=embeddingIntro, filterSize=5, nFilter=20, stride=3,
                                                   channels=self.vecDim)
            introEmbedding5 = tf.reduce_mean(
                layer.attention_layer(introEmbeddings5, hiddenUnits=hiddenNums, name='introEmbedding5'), axis=1)

            introEmbeddings8 = layer.conv_1d_layer(value=embeddingIntro, filterSize=8, nFilter=20, stride=4,
                                                   channels=self.vecDim)
            introEmbedding8 = tf.reduce_mean(
                layer.attention_layer(introEmbeddings8, hiddenUnits=hiddenNums, name='introEmbedding8'), axis=1)

            introAllInfo = tf.concat([introEmbedding3, introEmbedding5, introEmbedding8], axis=1)

            allInfo = tf.concat([titleEmbedding, tagEmbedding, introAllInfo], axis=1)
            # 层级预测结构

            l1hidden1 = layer.dense_layer(allInfo, weightShape=[hiddenNums * 7, self.numLabelsLevel1 * 10],
                                          outputShape=[self.numLabelsLevel1 * 10], name='l1hidden1')
            l1hidden2 = layer.dense_layer(l1hidden1, weightShape=[self.numLabelsLevel1 * 10, self.numLabelsLevel1*5],
                                          outputShape=[self.numLabelsLevel1*5], name='l1hidden2')
            l1Predict = layer.dense_layer(l1hidden2,
                                          weightShape=[self.numLabelsLevel1*5, self.numLabelsLevel1],
                                            outputShape=[self.numLabelsLevel1], active='sigmoid',
                                            name="L1pred")
            lossL1 = layer.square_loss(l1Predict, y1) #+ layer.cost_loss_multi(l1Predict, y1, weightOne=3.0)

            l2hidden1 = layer.dense_layer(tf.concat([allInfo, l1Predict, l1Predict], axis=1),
                                          weightShape=[hiddenNums * 7 + self.numLabelsLevel1 * 2,
                                                       self.numLabelsLevel2 * 4],
                                          outputShape=[self.numLabelsLevel2 * 4], name="l2hidden1")
            l2hidden2 = layer.dense_layer(l2hidden1,
                                          weightShape=[self.numLabelsLevel2 * 4,
                                                       self.numLabelsLevel2 * 2],
                                          outputShape=[self.numLabelsLevel2 * 2], name="l2hidden2")
            l2Predict = layer.dense_layer(l2hidden2,
                                          weightShape=[self.numLabelsLevel2 * 2, self.numLabelsLevel2],
                                          outputShape=[self.numLabelsLevel2], active="sigmoid", name="L2pred")
            lossL2 = layer.square_loss(l2Predict, y2)
            optiL1 = layer.optimize_op(lossL1, name='adam', rate=0.00001)
            optiL2 = layer.optimize_op(lossL2, name='adam', rate=0.00001)
            return {
                'l1pred': l1Predict, 'l2pred': l2Predict,
                'lossL1': lossL1, 'lossL2': lossL2,
                'optiL1': optiL1, 'optiL2': optiL2
            }

    # X should be an albums features
    # X = [
    # [ tags] , [titles], [intros]
    # ]
    def train(self, X, Y, isFirstTrain=True, batch_size=50, epoch=200, startStep=0):
        batchTags, _, batchNum = split2Batches(batch_size, X[0], None)
        tp.batchPadding(batchTags)
        batchTitles, _, batchNum = split2Batches(batch_size, X[1], None)
        tp.batchPadding(batchTitles)
        batchIntros, _, batchNum = split2Batches(batch_size, X[2], None)
        tp.batchPadding(batchIntros, maxPaddingLen=150)
        batchY1, _, batchNum = split2Batches(batch_size, Y[0], None)
        batchY2, _, batchNum = split2Batches(batch_size, Y[1], None)
        logModel = ''
        if isFirstTrain:
            logModel = 'w'
        else:
            logModel = 'a'
        logFile = open('./optimizerLog', logModel)
        logFile.write('TagTitleLoss\t|\tIntroLoss\t|\tFinalLoss\n')
        with tf.Session() as sess:
            if isFirstTrain:
                self.net = self.createModel()
                sess.run(tf.global_variables_initializer())
            else:
                # return {'tagTitleOP': optiTagTitle, 'introOP': optiIntro, 'finalOP': optiFinal,
                #         'predIntro': introPredict, 'predMixed': mixedPredict, 'finalPred': finalPredict,
                #         'introLoss': lossIntro, 'lossMixed': tagTitleLoss, 'finalLoss': finalLoss}
                self.readModel(sess)
                self.net = {}

                # optimizers
                self.net['optiL1'] = tf.get_default_graph().get_operation_by_name(self.opByName['optiL1'])
                self.net['optiL2'] = tf.get_default_graph().get_operation_by_name(self.opByName['optiL2'])

                # predicts
                self.net['l1pred'] = tf.get_default_graph().get_tensor_by_name(self.tensorsByName['l1pred'])
                self.net['l2pred'] = tf.get_default_graph().get_tensor_by_name(self.tensorsByName['l2pred'])

                # losses
                self.net['lossL1'] = tf.get_default_graph().get_tensor_by_name(self.tensorsByName['lossL1'])
                self.net['lossL2'] = tf.get_default_graph().get_tensor_by_name(self.tensorsByName['lossL2'])

            summary = tf.summary.FileWriter(self.tensorBoardPath)
            summary.add_graph(sess.graph, global_step=1)

            obop = tf.summary.merge_all()
            tag = tf.get_default_graph().get_tensor_by_name(self.tensorsByName['tag'])
            title = tf.get_default_graph().get_tensor_by_name(self.tensorsByName['title'])
            intro = tf.get_default_graph().get_tensor_by_name(self.tensorsByName['intro'])
            y1 = tf.get_default_graph().get_tensor_by_name(self.tensorsByName['labelLevel1'])
            y2 = tf.get_default_graph().get_tensor_by_name(self.tensorsByName['labelLevel2'])

            for step in range(epoch):
                epochSta = [0] * 3
                for k in range(batchNum):
                    # 'l1pred':l1Predict, 'l2pred':l2Predict,
                    # 'lossL1':lossL1, 'lossL2':lossL2,
                    # 'optiL1':optiL1, 'optiL2':optiL2
                    res = sess.run(
                        [self.net['optiL1'],  self.net['optiL2'], self.net['lossL1'], self.net['lossL2']
                         ],
                        feed_dict={tag: batchTags[k], title: batchTitles[k], intro: batchIntros[k], y1: batchY1[k],
                                   y2: batchY2[k]})

                    epochSta[0] += res[-2]
                    epochSta[1] += res[-1]
                epochSta = [str(i / batchNum) for i in epochSta]
                logFile.write('\t|\t'.join(epochSta) + '\n')
                logFile.flush()
                if (startStep + step) % 5 == 0:
                    ops = sess.run(obop, feed_dict={tag: batchTags[0], title: batchTitles[0], intro: batchIntros[0],
                                                    y1: batchY1[0], y2: batchY2[0]})

                    summary.add_summary(ops, startStep + step)
                if (startStep + step + 1) % 50 == 0:
                    self.saveModel(sess)
            if (epoch + startStep) % 50 != 0:
                self.saveModel(sess)
            logFile.close()
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
        with tf.Session() as sess:
            saver = self.readModel(sess)
            l1Predict = tf.get_default_graph().get_tensor_by_name(self.tensorsByName['l1pred'])
            l2Predict = tf.get_default_graph().get_tensor_by_name(self.tensorsByName['l2pred'])
            tag = tf.get_default_graph().get_tensor_by_name(self.tensorsByName['tag'])
            title = tf.get_default_graph().get_tensor_by_name(self.tensorsByName['title'])
            intro = tf.get_default_graph().get_tensor_by_name(self.tensorsByName['intro'])
            res = sess.run([l1Predict, l2Predict], feed_dict={tag: X[0], title: X[1], intro: X[2]})
            return res

    def outputGraph(self, file='./Record/'):
        with tf.Session() as sess:
            self.readModel(sess)
            graph = tf.graph_util.convert_variables_to_constants(sess=sess, input_graph_def=sess.graph_def,
                                                                 output_node_names=[
                                                                     self.tensorsByName['l1pred'][0:-2],
                                                                     self.tensorsByName['l2pred'][0:-2],
                                                                     self.tensorsByName['tag'][0:-2],
                                                                     self.tensorsByName['title'][0:-2],
                                                                     self.tensorsByName['intro'][0:-2]
                                                                 ])
            tf.train.write_graph(graph, file, 'graph.pb', as_text=False)
