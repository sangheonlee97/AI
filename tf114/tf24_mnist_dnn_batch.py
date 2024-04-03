import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
tf.compat.v1.set_random_seed(777)
from tensorflow.keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(y_train.shape)
print(y_test.shape)

X_train = X_train.reshape(-1, 28 * 28).astype('float32')/255.
X_test = X_test.reshape(-1, 28 * 28).astype('float32')/255.

batch_size = 32
epochs = 50

X = tf.compat.v1.placeholder(tf.float32, shape=(None, 28*28))
y = tf.compat.v1.placeholder(tf.float32, shape=(None, 10))

w1 = tf.compat.v1.get_variable('w1', shape=(28*28, 32), initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.compat.v1.Variable(tf.zeros([32]))
layer1 = tf.compat.v1.matmul(X, w1) + b1
layer1 = tf.compat.v1.nn.relu(layer1)

w2 = tf.compat.v1.get_variable('w2', shape=(32, 16), initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.compat.v1.Variable(tf.zeros([16]))
layer2 = tf.compat.v1.matmul(layer1, w2) + b2
layer2 = tf.compat.v1.nn.relu(layer2)

w3 = tf.compat.v1.get_variable('w3', shape=(16, 10), initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.compat.v1.Variable(tf.zeros([10]))
layer3 = tf.compat.v1.matmul(layer2, w3) + b3
hypothesis = tf.compat.v1.nn.softmax(layer3)

loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis + 1e-7), axis=1))  # categorical crossentropy
train = tf.compat.v1.train.AdamOptimizer(learning_rate=0.001).minimize(loss=loss)

with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    for step in range(epochs):
        avg = 0
        for i in range(len(X_train) // batch_size):
            _, c = sess.run([train, loss], feed_dict={X:X_train[i * batch_size : (i+1) * batch_size], y:y_train[i * batch_size : (i+1) * batch_size]})
            avg += c / (len(X_train) // batch_size)
        sess.run([train, loss], feed_dict={X:X_train[(len(X_train) // batch_size) * batch_size:], y:y_train[(len(X_train) // batch_size) * batch_size:]})
        print("step : ", step, "loss : ", avg)
    predict = sess.run(hypothesis, feed_dict={X:X_test})
from sklearn.metrics import accuracy_score
import numpy as np
test = np.argmax(y_test, axis=1)
pred = np.argmax(predict, axis=1)
acc = accuracy_score(test, pred)
print('acc : ', acc)
