import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler, MinMaxScaler
tf.compat.v1.set_random_seed(777)

X_data, y_data = fetch_california_housing(return_X_y=True)
y_data = np.array(y_data).reshape(-1, 1)

ss = MinMaxScaler()
X_data = ss.fit_transform(X_data)

X = tf.compat.v1.placeholder(tf.float32, shape=(None,X_data.shape[1]))
y = tf.compat.v1.placeholder(tf.float32, shape=(None,y_data.shape[1]))
w = tf.compat.v1.Variable(tf.compat.v1.random_normal(shape=(X_data.shape[1], 10)), name='weight')
b = tf.compat.v1.Variable(tf.compat.v1.zeros(shape=(1, 10)), name='bias')

layer = tf.nn.relu(tf.compat.v1.matmul(X, w) + b)

w2 = tf.compat.v1.Variable(tf.compat.v1.random_normal(shape=(10, y_data.shape[1])), name='weight')
b2 = tf.compat.v1.Variable(tf.compat.v1.zeros(shape=(1, y_data.shape[1])), name='bias')

hypothesis = tf.compat.v1.matmul(layer, w2) + b2
loss = tf.reduce_mean(tf.square(y - hypothesis))
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=5e-3)
train = optimizer.minimize(loss)

with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    for i in range(10000):
        _, vloss = sess.run([train, loss], feed_dict={X:X_data, y:y_data})
        print('i : ', i,'   loss : ', vloss)
    pred = sess.run(hypothesis, feed_dict={X:X_data})

from sklearn.metrics import r2_score
acc = r2_score(y_data, pred)
print('r2 : ', acc)