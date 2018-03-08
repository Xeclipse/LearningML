# coding:utf-8

import tensorflow as tf
from tensorflow.contrib.rnn import MultiRNNCell, BasicRNNCell, LSTMCell
import NeuralNetworkUtils.layers.layers as layer
import numpy as np
from NeuralNetworkUtils.utils.utils import oneHotLabels, split2Batches
from tensorFlowTest.libs.utils import weight_variable
from utils import textProcess as tp


class ComplexAlbumTextClassifier:
    def __init__(self):
        self.displayStep = 1
        self.numUnits = 21
        self.vocabDim = None
        self.vecDim = 10
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
            'predMixed': self.graphName + '/dense/mixedPredict:0',
            'finalPred': self.graphName + '/dense_2/finalPred:0',
            'predIntro': self.graphName + '/dense_1/introPred:0',#
            'introLoss': self.graphName + '/add_1:0',
            'lossMixed': self.graphName + '/add:0',
            'finalLoss': self.graphName + '/loss_4/cost_loss:0',
        }
        self.opByName = {
            "tagTitleOP": self.graphName + '/optimizer/adam_optimizer',
            "introOP": self.graphName + '/optimizer_1/adam_optimizer',
            "finalOP": self.graphName + '/optimizer_2/adam_optimizer',
        }

    def createModel(self):
        hiddenNums = 100
        with tf.name_scope(self.graphName):
            x1 = tf.placeholder(dtype=tf.int32, shape=[None, None], name="inputsTag")
            x2 = tf.placeholder(dtype=tf.int32, shape=[None, None], name="inputsTitle")
            x3 = tf.placeholder(dtype=tf.int32, shape=[None, None], name="inputsIntro")
            y = tf.placeholder(dtype=tf.float32, shape=[None, self.numLabels], name="outputs")
            intialembedding = tf.random_normal(shape=[self.vocabDim, self.vecDim], mean=0.0, stddev=0.1)
            embeddingTable = tf.Variable(intialembedding, name='embeddingTable')
            tf.summary.histogram(name='embedding', values=embeddingTable)

            # process titles
            embeddingTitle = tf.nn.embedding_lookup(embeddingTable, x2, max_norm=1.0)
            titleEmbeddings = layer.conv_1d_layer(embeddingTitle, filterSize=2, nFilter=20, stride=1,
                                                  channels=self.vecDim)
            # name="titleEmbedding")
            titleEmbedding = tf.reduce_mean(
                layer.attention_layer(value=titleEmbeddings, hiddenUnits=hiddenNums, name="titleEmbedding"), axis=1)

            # process tags
            embeddingTag = tf.nn.embedding_lookup(embeddingTable, x1, max_norm=1.0)
            tagEmbeddings2 = layer.conv_1d_layer(embeddingTag, filterSize=2, nFilter=20, stride=1, channels=self.vecDim)
            tagEmbedding2 = tf.reduce_mean(
                layer.attention_layer(value=tagEmbeddings2, hiddenUnits=hiddenNums, name="tagsEmbedding2"), axis=1)
            tagEmbeddings3 = layer.conv_1d_layer(embeddingTag, filterSize=3, nFilter=20, stride=1, channels=self.vecDim)
            tagEmbedding3 = tf.reduce_mean(
                layer.attention_layer(value=tagEmbeddings3, hiddenUnits=hiddenNums, name="tagsEmbedding3"), axis=1)
            tagEmbeddings4 = layer.conv_1d_layer(embeddingTag, filterSize=4, nFilter=20, stride=2, channels=self.vecDim)
            tagEmbedding4 = tf.reduce_mean(
                layer.attention_layer(value=tagEmbeddings4, hiddenUnits=hiddenNums, name="tagsEmbedding4"), axis=1)
            tagEmbedding = tf.concat([tagEmbedding2, tagEmbedding3, tagEmbedding4], axis=1)

            concat = tf.concat([titleEmbedding, tagEmbedding], 1)  # introEmbedding
            mixedPredict = layer.dense_layer(concat, weightShape=[hiddenNums*4, self.numLabels],
                                             outputShape=[self.numLabels], name="mixedPredict")

            tagTitleLoss = layer.cost_loss_multi(mixedPredict, y) + layer.square_loss(mixedPredict, y)
            optiTagTitle = layer.optimize_op(tagTitleLoss, name='adam', rate=0.0001)

            # process intross
            embeddingIntro = tf.nn.embedding_lookup(embeddingTable, x3, max_norm=1.0)
            introEmbeddings3 = layer.conv_1d_layer(value=embeddingIntro, filterSize=3, nFilter=50, stride=2,
                                                   channels=self.vecDim)
            introEmbedding3 = tf.reduce_mean(
                layer.attention_layer(introEmbeddings3, hiddenUnits=hiddenNums, name='introEmbedding3'), axis=1)

            introEmbeddings5 = layer.conv_1d_layer(value=embeddingIntro, filterSize=5, nFilter=50, stride=3,
                                                   channels=self.vecDim)
            introEmbedding5 = tf.reduce_mean(
                layer.attention_layer(introEmbeddings5, hiddenUnits=hiddenNums, name='introEmbedding5'), axis=1)

            introEmbeddings8 = layer.conv_1d_layer(value=embeddingIntro, filterSize=8, nFilter=50, stride=4,
                                                   channels=self.vecDim)
            introEmbedding8 = tf.reduce_mean(
                layer.attention_layer(introEmbeddings8, hiddenUnits=hiddenNums, name='introEmbedding8'), axis=1)

            introAllInfo = tf.concat([introEmbedding3, introEmbedding5, introEmbedding8], axis=1)
            introPredict = layer.dense_layer(introAllInfo, weightShape=[hiddenNums * 3, self.numLabels],
                                             outputShape=[self.numLabels], name="introPred")
            lossIntro = layer.square_loss(introPredict, y) + layer.cost_loss_multi(introPredict, y)
            optiIntro = layer.optimize_op(lossIntro, name='adam', rate=0.00001)

            finalPredict = layer.dense_layer(tf.concat([mixedPredict, introPredict], axis=1),
                                             weightShape=[self.numLabels * 2, self.numLabels],
                                             outputShape=[self.numLabels], name='finalPred', active='sigmoid')
            finalLoss = layer.cost_loss_multi(finalPredict, y) + layer.square_loss(finalPredict,y)
            optiFinal = layer.optimize_op(finalLoss, name='adam', rate=0.00001)

            return {'tagTitleOP': optiTagTitle, 'introOP': optiIntro, 'finalOP': optiFinal,
                    'predIntro': introPredict, 'predMixed': mixedPredict, 'finalPred': finalPredict,
                    'introLoss': lossIntro, 'lossMixed': tagTitleLoss, 'finalLoss': finalLoss}

    # X should be an albums features
    # X = [
    # [ tags] , [titles], [intros]
    # ]
    def train(self, X, Y, isFirstTrain=True, batch_size=50, epoch=200, startStep=0):
        batchTags, batchY, batchNum = split2Batches(batch_size, X[0], Y)
        tp.batchPadding(batchTags)
        batchTitles, batchY, batchNum = split2Batches(batch_size, X[1], Y)
        tp.batchPadding(batchTitles)
        batchIntros, batchY, batchNum = split2Batches(batch_size, X[2], Y)
        tp.batchPadding(batchIntros, maxPaddingLen=150)
        logModel =''
        if isFirstTrain:
            logModel = 'w'
        else:
            logModel = 'a'
        logFile = open('./optimizerLog', logModel)
        logFile.write('TagTitleLoss\t|\tIntroLoss\t|\tFinalLoss\n')
        with tf.Session() as sess:
            if isFirstTrain:
                self.net = self.createModel()
            else:
                # return {'tagTitleOP': optiTagTitle, 'introOP': optiIntro, 'finalOP': optiFinal,
                #         'predIntro': introPredict, 'predMixed': mixedPredict, 'finalPred': finalPredict,
                #         'introLoss': lossIntro, 'lossMixed': tagTitleLoss, 'finalLoss': finalLoss}
                self.readModel(sess)
                self.net = {}

                # optimizers
                self.net['tagTitleOP'] = tf.get_default_graph().get_operation_by_name(self.opByName['tagTitleOP'])
                self.net['introOP'] = tf.get_default_graph().get_operation_by_name(self.opByName['introOP'])
                self.net['finalOP'] = tf.get_default_graph().get_operation_by_name(self.opByName['finalOP'])
                # predicts
                self.net['predIntro'] = tf.get_default_graph().get_tensor_by_name(self.tensorsByName['predIntro'])
                self.net['predMixed'] = tf.get_default_graph().get_tensor_by_name(self.tensorsByName['predMixed'])
                self.net['finalPred'] = tf.get_default_graph().get_tensor_by_name(self.tensorsByName['finalPred'])
                #losses
                self.net['introLoss'] = tf.get_default_graph().get_tensor_by_name(self.tensorsByName['introLoss'])
                self.net['lossMixed'] = tf.get_default_graph().get_tensor_by_name(self.tensorsByName['lossMixed'])
                self.net['finalLoss'] = tf.get_default_graph().get_tensor_by_name(self.tensorsByName['finalLoss'])


            summary = tf.summary.FileWriter(self.tensorBoardPath)
            summary.add_graph(sess.graph, global_step=1)
            sess.run(tf.global_variables_initializer())
            obop = tf.summary.merge_all()
            tag = tf.get_default_graph().get_tensor_by_name(self.tensorsByName['tag'])
            title = tf.get_default_graph().get_tensor_by_name(self.tensorsByName['title'])
            intro = tf.get_default_graph().get_tensor_by_name(self.tensorsByName['intro'])
            y = tf.get_default_graph().get_tensor_by_name(self.tensorsByName['y'])
            sess.run(tf.global_variables_initializer())
            for step in range(epoch):
                epochSta = [0]*3
                for k in range(batchNum):
                    res = sess.run(
                        [self.net['tagTitleOP'], self.net['introOP'], self.net['finalOP'], self.net['lossMixed'],
                         self.net['introLoss'], self.net['finalLoss'],
                         ],
                        feed_dict={tag: batchTags[k], title: batchTitles[k], intro: batchIntros[k], y: batchY[k]})

                    epochSta[0] += res[3]
                    epochSta[1] += res[4]
                    epochSta[2] += res[5]
                epochSta = [str(i / batchNum) for i in epochSta]
                logFile.write('\t|\t'.join(epochSta)+'\n')
                if (startStep + step) % 5 == 0:
                    ops = sess.run(obop, feed_dict={tag: batchTags[0], title: batchTitles[0], intro: batchIntros[0],
                                                    y: batchY[0]})

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
            mixedPredict = tf.get_default_graph().get_tensor_by_name(self.tensorsByName['finalPred'])
            tag = tf.get_default_graph().get_tensor_by_name(self.tensorsByName['tag'])
            title = tf.get_default_graph().get_tensor_by_name(self.tensorsByName['title'])
            intro = tf.get_default_graph().get_tensor_by_name(self.tensorsByName['intro'])
            res = sess.run([mixedPredict], feed_dict={tag: X[0], title: X[1], intro: X[2]})
            return res
