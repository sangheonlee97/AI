import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
tf.compat.v1.set_random_seed(777)

X_data, y_data = load_breast_cancer(return_X_y=True)
print(X_data.shape)
print(y_data.shape)
# y_data = pd.get_dummies(y_data)
print(y_data)

ss = MinMaxScaler()
X_data = ss.fit_transform(X_data)

X = tf.compat.v1.placeholder(tf.float32, shape=(None,X_data.shape[1]))
y = tf.compat.v1.placeholder(tf.float32, shape=(None,))
w = tf.compat.v1.Variable(tf.compat.v1.random_normal(shape=(X_data.shape[1], 10)), name='weight')
b = tf.compat.v1.Variable(tf.compat.v1.zeros(shape=(1, 10)), name='bias')

layer = tf.compat.v1.matmul(X, w) + b

w2 = tf.compat.v1.Variable(tf.compat.v1.random_normal(shape=(10, 1)), name='weight')
b2 = tf.compat.v1.Variable(tf.compat.v1.zeros(shape=(1, 1)), name='bias')

hypothesis = tf.nn.sigmoid(tf.compat.v1.matmul(layer, w2) + b2)
# loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis=1))  # categorical crossentropy
loss = -tf.reduce_mean(y * tf.math.log(hypothesis) + (1 - y) * tf.math.log(1-hypothesis))

optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=5e-5)
train = optimizer.minimize(loss)

with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    for i in range(10000):
        _, vloss = sess.run([train, loss], feed_dict={X:X_data, y:y_data})
        print('i : ', i,'   loss : ', vloss)
    pred = sess.run(hypothesis, feed_dict={X:X_data}).round()


from sklearn.metrics import accuracy_score
acc = accuracy_score(y_data, pred)
print('acc : ', acc)