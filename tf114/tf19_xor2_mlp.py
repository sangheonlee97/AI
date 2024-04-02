import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
tf.compat.v1.set_random_seed(777)

X_data = [[0, 0], [0, 1], [1, 0], [1, 1]]
y_data = [[0], [1], [1], [0]]

X = tf.compat.v1.placeholder(tf.float32, shape=(None, 2))
y = tf.compat.v1.placeholder(tf.float32, shape=(None, 1))


w1 = tf.compat.v1.Variable(tf.random.normal((2, 20)), dtype=tf.float32)
b1 = tf.compat.v1.Variable(tf.zeros([20]), dtype=tf.float32)
hypothesis = tf.compat.v1.nn.relu(tf.compat.v1.matmul(X, w1) + b1)

w2 = tf.compat.v1.Variable(tf.random.normal((20, 20)), dtype=tf.float32)
b2 = tf.compat.v1.Variable(tf.zeros([20]), dtype=tf.float32)
hypothesis2 = tf.compat.v1.nn.relu(tf.compat.v1.matmul(hypothesis, w2) + b2)

w3 = tf.compat.v1.Variable(tf.random.normal((20, 20)), dtype=tf.float32)
b3 = tf.compat.v1.Variable(tf.zeros([20]), dtype=tf.float32)
hypothesis3 = tf.compat.v1.nn.relu(tf.compat.v1.matmul(hypothesis2, w3) + b3)

w4 = tf.compat.v1.Variable(tf.random.normal((20, 20)), dtype=tf.float32)
b4 = tf.compat.v1.Variable(tf.zeros([20]), dtype=tf.float32)
hypothesis4 = tf.compat.v1.nn.sigmoid(tf.compat.v1.matmul(hypothesis3, w4) + b4)

w5 = tf.compat.v1.Variable(tf.random.normal((20, 1)), dtype=tf.float32)
b5 = tf.compat.v1.Variable(tf.zeros([1]), dtype=tf.float32)
hypothesis5 = tf.compat.v1.nn.sigmoid(tf.compat.v1.matmul(hypothesis4, w5) + b5)


loss = -tf.reduce_mean(y * tf.math.log(hypothesis5) + (1 - y) * tf.math.log(1 - hypothesis5))
train = tf.compat.v1.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

predicted = tf.cast(hypothesis5 > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype=tf.float32))

epochs = 1000
with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    for i in range(epochs):
        _, loss_, acc = sess.run([train, loss, accuracy], feed_dict={X:X_data, y:y_data})
        print('loss : ', loss_)
    pred = sess.run(hypothesis5, feed_dict={X:X_data}).round()
from sklearn.metrics import accuracy_score
# acc = accuracy_score(y_data, pred)
print(pred)
print('acc : ', acc)