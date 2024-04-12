import tensorflow as tf

import numpy as np
tf.compat.v1.set_random_seed(777)

from tensorflow.keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255.
X_test = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255.

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

X = tf.compat.v1.placeholder(tf.float32, (None, 28, 28, 1))
y = tf.compat.v1.placeholder(tf.float32, (None, 10))

w1 = tf.compat.v1.get_variable('w1', shape=(3, 3, 1, 64))
b1 = tf.compat.v1.Variable(tf.zeros(shape=(1, 64)))
L1 = tf.compat.v1.nn.conv2d(X, w1, strides=(1,1,1,1), padding='VALID') + b1
L1 = tf.nn.relu(L1)
L1_MaxPool = tf.nn.max_pool2d(L1, (1, 2, 2, 1), strides=(1, 2, 2, 1), padding='VALID')

w2 = tf.compat.v1.get_variable('w2', shape=(3, 3, 64, 32))
L2 = tf.compat.v1.nn.conv2d(L1_MaxPool, w2, strides=(1,1,1,1), padding='VALID')

flatten = tf.compat.v1.reshape(L2, (-1, L2.shape[1] * L2.shape[2] * L2.shape[3]))

w3 = tf.compat.v1.get_variable('w3', shape=(flatten.shape[1], 32), dtype=tf.float32)
L3 = tf.compat.v1.matmul(flatten, w3)

w4 = tf.compat.v1.get_variable('w4', (32, 10), tf.float32)
hypothesis = tf.compat.v1.nn.softmax(tf.compat.v1.matmul(L3, w4))

loss = tf.reduce_mean(-tf.reduce_sum(y * tf.math.log(hypothesis + 1e-7), axis=1))  # categorical crossentropy
train = tf.compat.v1.train.AdamOptimizer().minimize(loss)

EPOCHS = 3
batch_size = 64
with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    for epoch in range(EPOCHS):
        avg = 0
        for i in range(len(X_train) // batch_size):
            _, c = sess.run([train, loss], feed_dict={X:X_train[i * batch_size : (i+1) * batch_size], y:y_train[i * batch_size : (i+1) * batch_size]})
            avg += c / (len(X_train) // batch_size)
            print('epoch : ', epoch + 1, " |  batch : ", i + 1, " |  loss : ", avg)
        sess.run([train, loss], feed_dict={X:X_train[(len(X_train) // batch_size) * batch_size:], y:y_train[(len(X_train) // batch_size) * batch_size:]})
        print("epoch : ", epoch + 1, "loss : ", avg)
    predict = sess.run(hypothesis, feed_dict={X:X_test})
from sklearn.metrics import accuracy_score
test = np.argmax(y_test, axis=1)
pred = np.argmax(predict, axis=1)
acc = accuracy_score(test, pred)
print('acc : ', acc)
