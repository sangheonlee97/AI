import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
tf.compat.v1.set_random_seed(777)

X_data = [[0, 0], [0, 1], [1, 0], [1, 1]]
y_data = [[0], [1], [1], [0]]

X = tf.compat.v1.placeholder(tf.float32, shape=(None, 2))
y = tf.compat.v1.placeholder(tf.float32, shape=(None, 1))
w = tf.compat.v1.Variable(tf.random.normal((2, 1)), dtype=tf.float32)
b = tf.compat.v1.Variable(tf.zeros([1]), dtype=tf.float32)

# hypothesis = tf.compat.v1.matmul(X, w) + b
hypothesis = tf.compat.v1.nn.sigmoid(tf.compat.v1.matmul(X, w) + b)
# loss = tf.reduce_mean(tf.square(y - hypothesis))
loss = -tf.reduce_mean(y * tf.math.log(hypothesis) + (1 - y) * tf.math.log(1-hypothesis))
train = tf.compat.v1.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
epochs = 1000
with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    for i in range(epochs):
        _, loss_ = sess.run([train, loss], feed_dict={X:X_data, y:y_data})
    pred = sess.run(hypothesis, feed_dict={X:X_data}).round()
from sklearn.metrics import accuracy_score
acc = accuracy_score(y_data, pred)
print('acc : ', acc)
print('loss : ', loss_)
print(pred)