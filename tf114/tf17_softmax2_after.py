import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
import numpy as np
tf.compat.v1.set_random_seed(777)

X_data = [
    [1,2,1,1],
    [2,1,3,2],
    [3,1,3,4],
    [4,1,5,5],
    [1,7,5,5],
    [1,2,5,6],
    [1,6,6,6],
    [1,7,6,7]
]
y_data = [
    [0,0,1],
    [0,0,1],
    [0,0,1],
    [0,1,0],
    [0,1,0],
    [0,1,0],
    [1,0,0],
    [1,0,0]
]

X = tf.compat.v1.placeholder(tf.float32, shape=(None,4))
y = tf.compat.v1.placeholder(tf.float32, shape=(None,3))
w = tf.compat.v1.Variable(tf.compat.v1.random_normal(shape=(4, 3)), name='weight')
b = tf.compat.v1.Variable(tf.compat.v1.zeros(shape=(1,3)), name='bias')

hypothesis = tf.nn.softmax(tf.compat.v1.matmul(X, w) + b)
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis=1))  # categorical crossentropy
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-4)
train = optimizer.minimize(loss)

with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    for i in range(60000):
        _, vloss = sess.run([train, loss], feed_dict={X:X_data, y:y_data})
        print('i : ', i,'   loss : ', vloss)
    pred = sess.run(hypothesis, feed_dict={X:X_data})
pred = np.array(pred)
arg_pred = np.zeros((8,3))
for i, v in enumerate(pred):
    arg_pred[i][np.argmax(pred[i])] = 1
print(arg_pred)

from sklearn.metrics import accuracy_score
acc = accuracy_score(y_data, arg_pred)
print('acc : ', acc)