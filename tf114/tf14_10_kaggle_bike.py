# 7. load_diabets
# 8. california
# 9. dacon ddarung
# 10. kaggle bike

import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
tf.compat.v1.set_random_seed(42)

path = "c:\\Study\\_data\\kaggle\\bike\\"

train_csv = pd.read_csv(path + "train.csv" , index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission_csv = pd.read_csv(path + "samplesubmission.csv")

X = train_csv.drop(['casual', 'registered', 'count'], axis=1) 
y = train_csv['count']

# print(X.shape)      #(10886, 8)
# print(y.shape)      #(10886)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=713)

mms = MinMaxScaler()
X_train = mms.fit_transform(X_train)
X_test = mms.transform(X_test)

Xp = tf.compat.v1.placeholder(tf.float32, shape=(None, X.shape[1]))
yp = tf.compat.v1.placeholder(tf.float32, shape=(None, ))

w = tf.compat.v1.Variable(tf.compat.v1.random_normal((X.shape[1], 1)))
b = tf.compat.v1.Variable(0, dtype=tf.float32)

hypothesis = tf.compat.v1.matmul(Xp, w) + b
loss = tf.compat.v1.reduce_mean(tf.compat.v1.square(yp - hypothesis))
opti = tf.compat.v1.train.AdamOptimizer(learning_rate=0.01)
train = opti.minimize(loss)

with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    for step in range(100):
        sess.run(train, feed_dict={Xp:X_train, yp:y_train})
    pred = sess.run(hypothesis, feed_dict={Xp:X_test})
    
r2 = r2_score(y_test, pred)
print('r2 : ', r2)