import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
tf.compat.v1.set_random_seed(777)

path = "../_data/dacon/ddarung/"

train_csv = pd.read_csv(path + "train.csv", index_col=0)        

train_csv['hour_bef_precipitation'] = train_csv['hour_bef_precipitation'].fillna(0)
train_csv['hour_bef_pm10'] = train_csv['hour_bef_pm10'].fillna(0)
train_csv['hour_bef_pm2.5'] = train_csv['hour_bef_pm2.5'].fillna(0)
train_csv['hour_bef_windspeed'] = train_csv['hour_bef_windspeed'].fillna(0)
train_csv['hour_bef_temperature'] = train_csv['hour_bef_temperature'].fillna(train_csv['hour_bef_temperature'].mean())
train_csv['hour_bef_humidity'] = train_csv['hour_bef_humidity'].fillna(train_csv['hour_bef_humidity'].mean())
train_csv['hour_bef_visibility'] = train_csv['hour_bef_visibility'].fillna(train_csv['hour_bef_visibility'].mean())
train_csv['hour_bef_ozone'] = train_csv['hour_bef_ozone'].fillna(train_csv['hour_bef_ozone'].mean())


X_data = train_csv.drop(['count'], axis=1)
y_data = train_csv['count']

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, shuffle=True, random_state=3)
y_train = np.array(y_train).reshape(-1, 1)
y_test = np.array(y_test).reshape(-1, 1)

ss = MinMaxScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

X = tf.compat.v1.placeholder(tf.float32, shape=(None,X_data.shape[1]))
y = tf.compat.v1.placeholder(tf.float32, shape=(None,y_train.shape[1]))
w = tf.compat.v1.Variable(tf.compat.v1.random_normal(shape=(X_data.shape[1], 10)), name='weight')
b = tf.compat.v1.Variable(tf.compat.v1.zeros(shape=(1, 10)), name='bias')

layer = tf.nn.relu(tf.compat.v1.matmul(X, w) + b)

w2 = tf.compat.v1.Variable(tf.compat.v1.random_normal(shape=(10, y_train.shape[1])), name='weight')
b2 = tf.compat.v1.Variable(tf.compat.v1.zeros(shape=(1, y_train.shape[1])), name='bias')

hypothesis = tf.compat.v1.matmul(layer, w2) + b2
loss = tf.reduce_mean(tf.square(y - hypothesis))
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=5e-3)
train = optimizer.minimize(loss)

with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    for i in range(10000):
        _, vloss = sess.run([train, loss], feed_dict={X:X_train, y:y_train})
        print('i : ', i,'   loss : ', vloss)
    pred = sess.run(hypothesis, feed_dict={X:X_test})

from sklearn.metrics import r2_score
acc = r2_score(y_test, pred)
print('r2 : ', acc)