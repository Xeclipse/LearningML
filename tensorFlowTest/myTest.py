import tensorflow as tf
import numpy as np

trainX = np.asanyarray(
    [
        [1, 2, 3, 4],
        [2, 3, 4, 5],
        [3, 2, 3, 4],
        [4, 2, 3, 4]
    ]
)
trainY = np.asanyarray(
    [
        [1],
        [1],
        [0],
        [0]
    ]
)

inputDim = 4
outputDim = 1
hiddenDim = 1


x = tf.placeholder(dtype=tf.float32, shape=[None, inputDim], name="inputs")
y = tf.placeholder(dtype=tf.float32, shape=[None, outputDim], name='pred')
w1 = tf.Variable(tf.random_normal(shape=[inputDim, hiddenDim]), name="weights1")
b1 = tf.Variable(tf.random_normal(shape=[hiddenDim]), name="bias1")
hiddenNode1 = tf.add(tf.matmul(x, w1), b1, name="layer1")

w2 = tf.Variable(tf.random_normal(shape=[hiddenDim, outputDim]), name="weights2")
b2 = tf.Variable(tf.random_normal(shape=[outputDim]), name="bias2")
ypred = tf.add(tf.matmul(hiddenNode1, w2), b2, name="pred")

loss = tf.reduce_mean(tf.pow(y - ypred, 2))
optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(1000):
        sess.run(optimizer, feed_dict={x: trainX, y: trainY})
        print sess.run(loss, feed_dict={x: trainX, y: trainY})
    print "\n"
    print sess.run(ypred, feed_dict={x: trainX})
