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


class MultiAlbumTextClassifier:
    def __init__(self):
        self.displayStep = 1
        self.numUnits = 21
        self.vocabDim = 12
        self.vecDim = 20
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
            'lossMixed': self.graphName + '/loss/square_loss:0',
            'standardLoss': self.graphName + '/loss_1/cost_loss:0',
        }
        self.opByName={
            "optimizer": self.graphName+'/optimizer/adam_optimizer'
        }

    def attention_layer(self, value, hiddenUnits, name):
        # todo:finish this network and build a class
        with tf.name_scope("attention_layer"):
            # y = tf.placeholder(dtype=tf.float32, shape=[None, self.numLabels], name="outputs"
            with tf.variable_scope("rnn_attention_" + name, reuse=False):
                with tf.variable_scope("rnn_encoder_variable", reuse=False):
                    cell = LSTMCell(num_units=hiddenUnits, initializer=tf.truncated_normal_initializer)
                    outputs, lastStates = layer.dynamicRnnLayer(value, cell)
                with tf.variable_scope("rnn_attention_variable", reuse=False):
                    attentionCell = LSTMCell(num_units=1, initializer=tf.truncated_normal_initializer)
                    attentions, lastAttentionState = layer.dynamicRnnLayer(outputs, attentionCell)
            batch = tf.shape(value)[0]
            attentionsRepeat = tf.ones(shape=[batch,1,hiddenUnits])
            attentionMatrix = tf.matmul( attentions,attentionsRepeat)
            attentionEmbeddings = tf.multiply(attentionMatrix, outputs)
            return attentionEmbeddings

    def createModel(self):
        with tf.name_scope(self.graphName):
            x1 = tf.placeholder(dtype=tf.int32, shape=[None, None], name="inputsTag")
            x2 = tf.placeholder(dtype=tf.int32, shape=[None, None], name="inputsTitle")
            y = tf.placeholder(dtype=tf.float32, shape=[None, self.numLabels], name="outputs")
            intialembedding = tf.random_normal(shape=[self.vocabDim, self.vecDim], mean=0.0, stddev=0.1)
            embeddingTable = tf.Variable(intialembedding,name='embeddingTable')
            tf.summary.histogram(name='embedding', values=embeddingTable)

            embeddingTag = tf.nn.embedding_lookup(embeddingTable, x1, max_norm=1.0)
            embeddingTitle = tf.nn.embedding_lookup(embeddingTable, x2, max_norm=1.0)
            titleEmbeddings = self.attention_layer(value=embeddingTitle, hiddenUnits=100,
                                                name="titleEmbedding")
            tagEmbeddings = self.attention_layer(value=embeddingTag, hiddenUnits=100,
                                               name="tagsEmbedding")


            titleEmbedding = tf.reduce_mean(titleEmbeddings, axis=1)
            tagEmbeddings2 = self.attention_layer(value=tagEmbeddings,hiddenUnits=100, name="higher_embedding")
            tagEmbedding = tf.reduce_mean(tagEmbeddings2, axis=1)


            concat = tf.concat([ titleEmbedding, tagEmbedding], 1)#introEmbedding
            mixedPredict = layer.dense_layer(concat, weightShape=[100 * 2, self.numLabels],
                                             outputShape=[self.numLabels], name="mixedPredict")

            standardLoss = layer.square_loss(mixedPredict, y)
            lossMixed = layer.cost_loss_multi(mixedPredict, y)+standardLoss
            optimizer = layer.optimize_op(lossMixed,name ='adam', rate=0.1)
            return {'optimizer': optimizer, 'lossMixed': lossMixed, 'predMixed':mixedPredict, 'standardLoss':standardLoss}




    # X should be an albums features
    # X = [
    # [ tags] , [titles], [intros]
    # ]
    def train(self, X, Y, isFirstTrain=True ,batch_size=1000, epoch=200, startStep= 0):
        batchTags, batchY, batchNum = split2Batches(batch_size, X[0], Y)
        tp.batchPadding(batchTags)
        batchTitles, batchY, batchNum = split2Batches(batch_size, X[1], Y)
        tp.batchPadding(batchTitles)
        # batchIntros, batchY, batchNum = split2Batches(batch_size, X[2], Y)
        # tp.batchPadding(batchIntros)
        with tf.Session() as sess:
            if isFirstTrain:
                self.net = self.createModel()
            else:
                self.readModel(sess)
                self.net = {}
                # return {'optimizer': optimizer, 'totalLoss': totalLoss, 'lossTags': lossTags, 'lossTitle': lossTitle,
                #     'lossIntro': lossIntro, 'lossMixed': lossMixed}
                self.net['optimizer']= tf.get_default_graph().get_operation_by_name(self.opByName['optimizer'])
                # self.net['totalLoss'] = tf.get_default_graph().get_tensor_by_name(self.tensorsByName['loss'])
                # self.net['lossTags'] = tf.get_default_graph().get_tensor_by_name(self.tensorsByName['lossTag'])
                # self.net['lossTitle'] = tf.get_default_graph().get_tensor_by_name(self.tensorsByName['lossTitle'])
                # self.net['lossIntro'] = tf.get_default_graph().get_tensor_by_name(self.tensorsByName['lossIntro'])
                self.net['lossMixed'] = tf.get_default_graph().get_tensor_by_name(self.tensorsByName['lossMixed'])
                self.net['predMixed'] = tf.get_default_graph().get_tensor_by_name(self.tensorsByName['predMixed'])
                self.net['standardLoss'] = tf.get_default_graph().get_tensor_by_name(self.tensorsByName['standardLoss'])
            summary = tf.summary.FileWriter(self.tensorBoardPath)
            summary.add_graph(sess.graph, global_step=1)
            sess.run(tf.global_variables_initializer())
            obop = tf.summary.merge_all()
            tag = tf.get_default_graph().get_tensor_by_name(self.tensorsByName['tag'])
            title = tf.get_default_graph().get_tensor_by_name(self.tensorsByName['title'])
            #intro = tf.get_default_graph().get_tensor_by_name(self.tensorsByName['intro'])
            y = tf.get_default_graph().get_tensor_by_name(self.tensorsByName['y'])
            sess.run(tf.global_variables_initializer())
            for step in range(epoch):
                epochSta = [0]*2
                for k in range(batchNum):
                    res = sess.run(
                        [self.net['optimizer'], self.net['lossMixed'],  self.net['standardLoss'],self.net['predMixed']],
                        feed_dict={tag: batchTags[k], title: batchTitles[k], y: batchY[k]})#intro: batchIntros[k],
                    epochSta[0] += res[1]
                    epochSta[1] += res[2]
                    # epochSta[2] += res[2]
                    # epochSta[3] += res[3]
                    # epochSta[4] += res[4]
                    # epochSta[5] += res[5]

                epochSta = [str(i / batchNum) for i in epochSta]
                # if i % self.displayStep == 0:
                print '\t|\t'.join(epochSta)
                ops = sess.run(obop, feed_dict={tag: batchTags[0], title: batchTitles[0],
                                                y: batchY[0]})#intro: batchIntros[0],

                summary.add_summary(ops, startStep+step)
                if (step+1)%100 ==0:
                    self.saveModel(sess)
            if (epoch)%100!=0:
                self.saveModel(sess)
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

    def predict(self, X ):
        with tf.Session() as sess:
            saver = self.readModel(sess)
            # tagPredict = tf.get_default_graph().get_tensor_by_name(self.tensorsByName['predTag'])
            # introPredict = tf.get_default_graph().get_tensor_by_name(self.tensorsByName['predTitle'])
            # titlePredict = tf.get_default_graph().get_tensor_by_name(self.tensorsByName['predIntro'])
            mixedPredict = tf.get_default_graph().get_tensor_by_name(self.tensorsByName['predMixed'])
            tag = tf.get_default_graph().get_tensor_by_name(self.tensorsByName['tag'])
            title = tf.get_default_graph().get_tensor_by_name(self.tensorsByName['title'])
            #intro = tf.get_default_graph().get_tensor_by_name(self.tensorsByName['intro'])
            res = sess.run([mixedPredict], feed_dict={tag: X[0], title: X[1]})#, intro: X[2]
            return res
# album = AlbumTextClassifier()
# album.createModel()
# X, Y, length = genToyData()
# Y = oneHotLabels(Y.tolist())
# print Y
# atten = AttentionClassifier()
# with tf.Session() as sess:
#     atten.train(X, Y)
#     sess.close()
