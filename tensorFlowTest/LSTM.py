import tensorflow as tf


initializer = tf.zeros_initializer((48,))

with tf.Session() as sess:
    sess.run(initializer)