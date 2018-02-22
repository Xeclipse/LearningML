import tensorflow as tf
from tensorFlowTest.libs.utils import weight_variable, bias_variable


def embedding_layer(value, vocabDim, vecDim):
    with tf.name_scope("embedding"):
        W = weight_variable([vocabDim, vecDim])
        embeddingVector = tf.nn.embedding_lookup(params=W, ids=value)
        tf.summary.histogram(name='embedding', values=W)
        return embeddingVector


def dense_layer(value, weightShape, outputShape, name=None, active= 'relu'):
    with tf.name_scope("dense"):
        weight = weight_variable(shape=weightShape)
        bias = bias_variable(shape=outputShape)
        if active=='relu':
            hidden = tf.nn.relu(tf.matmul(value, weight) + bias, name=name)
        else:
            hidden = tf.nn.sigmoid(tf.matmul(value, weight) + bias, name=name)
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


def optimize_op(value, name = 'adagard',rate = 0.01):
    with tf.name_scope("optimizer"):
        optimizer = None
        if name =='adagard':
            optimizer = tf.train.AdagradOptimizer(learning_rate=rate)
        if name =='adam':
            optimizer = tf.train.AdamOptimizer(learning_rate=rate)
        if name == 'adadelta':
            optimizer = tf.train.AdadeltaOptimizer(learning_rate=rate)
        if name == 'rmsprop':
            optimizer = tf.train.RMSPropOptimizer(learning_rate=rate)
        if name == 'adam':
            optimizer = tf.train.AdamOptimizer(learning_rate=rate)
        return optimizer.minimize(value, name=name+'_optimizer')

def optimize_op_rsrp(value, rate = 0.01):
    with tf.name_scope("optimizer"):
        rsrp = tf.train.RMSPropOptimizer(learning_rate=rate)
        return rsrp.minimize(value, name='rsrp_optimizer')

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

def cost_loss_multi(ypred, y, weightOne =5.0):
    with tf.name_scope("loss"):
        sub = ypred - y
        weight =  tf.scalar_mul(weightOne, y)
        weightPow2 = tf.multiply(tf.multiply(sub, sub), weight)
        L2 = tf.reduce_sum(input_tensor=weightPow2, axis=1)
        loss = tf.reduce_mean(L2, name='cost_loss')
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
        dim = int(tf.math.ceil(1.0 * sendim / stride)) * nFilter
        flat = flatten(value=hidden, flattenShape=[-1, dim])
        dense = dense_layer(value=flat, weightShape=[dim, outdim], outputShape=[outdim])
        return dense

def square_loss(ypred, y):
    with tf.name_scope("loss"):
        cost = tf.reduce_mean(tf.reduce_sum(tf.multiply( (y-ypred), (y-ypred)), axis=1), name="square_loss")
        tf.summary.scalar(name="square_loss", tensor=cost)
        return cost

def dynamicRnnLayer(value, cell, lengths = None):
    with tf.name_scope("RNN"):
        if lengths is not None:
            return tf.nn.dynamic_rnn(cell=cell, inputs=value, sequence_length=lengths, dtype=tf.float32)
        else:
            return tf.nn.dynamic_rnn(cell=cell, inputs=value, dtype=tf.float32)