import tensorflow as tf

tf.set_random_seed(123)
sess = tf.compat.v1.Session()

X_data = [1, 2, 3, 4, 5]
y_data = [3, 5, 7, 9, 11]
X = tf.compat.v1.placeholder(dtype=tf.float32)
y = tf.compat.v1.placeholder(dtype=tf.float32)

w = tf.compat.v1.Variable(tf.random_normal([1]), dtype=tf.float32)
b = tf.compat.v1.Variable(tf.random_normal([1]), dtype=tf.float32)


hypothesis = X * w + b

loss = tf.compat.v1.reduce_mean(tf.square(hypothesis - y))
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.001)
train = optimizer.minimize(loss=loss)

sess.run(tf.compat.v1.global_variables_initializer())

for step in range(10000):
    _, loss_val, w_val, b_val = sess.run([train, loss, w, b], feed_dict={X : X_data , y : y_data })
    print("%-5s" % step, '\t',  "%-12s" % loss_val,'\t',  "%-10s" % w_val[0],'\t', "%-12s" %  b_val[0])
    

X_pred = [6, 7, 8]
X_test = tf.compat.v1.placeholder(tf.float32, shape=[None])
y_pred = X_test * w_val + b_val
predict = sess.run(y_pred, feed_dict={X_test : X_pred})
print('[6, 7, 8]의 예측값 : [{}, {}, {}]'.format(predict[0], predict[1], predict[2]))

sess.close()