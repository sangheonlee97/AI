import tensorflow as tf
import numpy as np
tf.compat.v1.set_random_seed(123)

X_train = np.array([[[[1], [2], [3]],
                     [[4], [5], [6]],
                     [[7], [8], [9]]]])
print(X_train.shape)    # (1, 3, 3, 1)

X = tf.compat.v1.placeholder(tf.float32, (None, 3, 3, 1))
w = tf.compat.v1.constant([[[[1.]], [[0.]]],
                           [[[1.]], [[0.]]]])
print(w)

L1 = tf.nn.conv2d(X, w, strides=(1, 1, 1, 1), padding='VALID')
print(L1)

sess = tf.compat.v1.Session()
output = sess.run(L1, feed_dict={X:X_train})
print(output)
print(output.shape)
sess.close()