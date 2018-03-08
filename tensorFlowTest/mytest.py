import tensorflow as tf



X= [
    [[1.0],[0.0]],
    [[0.0],[2.0]]
]

Y = [
    [[1.0,2.0],[1.0,2.0]],
    [[1.0,2.0],[1.0,2.0]]
]


x = tf.placeholder(shape=[2,2,1], dtype=tf.float32)
y = tf.placeholder(shape=[2,2,2], dtype=tf.float32)
ans = tf.multiply(x,y)
norm = tf.nn.l2_normalize(X,dim=1)
with tf.Session() as sess:
    print sess.run(norm, feed_dict={x:X,y:Y})