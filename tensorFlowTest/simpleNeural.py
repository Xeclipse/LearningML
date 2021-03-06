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



def simpleNueralNetWork(inputDim, outputDim, hiddenDim=1):
    with tf.name_scope("simpleNeural"):
        x = tf.placeholder(dtype=tf.float32, shape=[None, inputDim], name="inputs")
        y = tf.placeholder(dtype=tf.float32, shape=[None, outputDim], name='pred')
        with tf.variable_scope("dense"):
            w1 = tf.Variable(tf.random_normal(shape=[inputDim, hiddenDim]), name="weights1")
            b1 = tf.Variable(tf.random_normal(shape=[hiddenDim]), name="bias1")
            w2 = tf.Variable(tf.random_normal(shape=[hiddenDim, outputDim]), name="weights2")
            b2 = tf.Variable(tf.random_normal(shape=[outputDim]), name="bias2")

        with tf.name_scope("layer"):
            hiddenNode1 = tf.add(tf.matmul(x, w1), b1, name="dense")
            ypred = tf.add(tf.matmul(hiddenNode1, w2), b2, name="predict")
            loss = tf.reduce_mean(tf.pow(y - ypred, 2), name="loss")
            optimizer = tf.train.GradientDescentOptimizer(0.01, name="optimizer").minimize(loss)

        return {"optimizer": optimizer, "loss": loss, "predict": ypred}


with tf.Session() as sess:
    net = simpleNueralNetWork(4, 1)
    writer = tf.summary.FileWriter('/Users/nali/PycharmProjects/LearningML/tensorFlowTest/tensorBoard/dict')
    writer.add_graph(graph=sess.graph)
    x = tf.get_default_graph().get_tensor_by_name("simpleNeural/inputs:0")
    y = tf.get_default_graph().get_tensor_by_name("simpleNeural/pred:0")
    sess.run(tf.global_variables_initializer())
    for i in range(10):
        sess.run(net["optimizer"], feed_dict={x: trainX, y: trainY})
        ans = sess.run(net["loss"], feed_dict={x: trainX, y: trainY})
        print ans
    ans = sess.run(net["predict"],  feed_dict={x: trainX})
    print ans
    saver = tf.train.Saver()
    saver.save(sess,'./save/model.nn')

    graph = tf.graph_util.convert_variables_to_constants(sess = sess, input_graph_def=sess.graph_def, output_node_names=["simpleNeural/layer/predict"])
    tf.train.write_graph(graph, './', 'graph.pb', as_text=False)

    sess.close()


# saver = tf.train.import_meta_graph("./save/model.nn.meta")
# with tf.Session() as sess:
#     saver.restore(sess, "./save/model.nn")
#     predict=tf.get_default_graph().get_tensor_by_name("predict:0")
#     input=tf.get_default_graph().get_tensor_by_name("inputs:0")
#     ans = sess.run(predict, feed_dict={input:trainX})
#     sess.close()
#     with open("./testOut",'w') as fot:
#         for i in ans:
#             for t in i:
#                 fot.write(str(t))
#                 fot.write(' ')
#             fot.write('\n')
