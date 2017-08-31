import tensorflow as tf

x = tf.constant(2)
y = tf.constant(5)


def f1(): return tf.multiply(x, 17)


def f2(): return tf.add(y, 23)


r = tf.cond(tf.less(x, y), lambda: (x,y), lambda:  (y,x))

with tf.Session() as sess:
    print sess.run(r)
    # r is set to f1().
    # Operations in f2 (e.g., tf.add) are not executed.
