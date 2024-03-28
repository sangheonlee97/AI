import tensorflow as tf
tf.compat.v1.set_random_seed(42)

X_data = [[73., 51., 65.],
          [92., 98., 11.],
          [89., 31., 33.],
          [99., 33., 100.],
          [17., 66., 79.]]
y_data = [[152.], [185.], [180.], [205.], [142.]]

X = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None,3))
y = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None,1))

w = tf.compat.v1.Variable(tf.compat.v1.random_normal((3, 1)), dtype=tf.float32)
b = tf.compat.v1.Variable([1], dtype=tf.float32)

# hypothesis = X * w + b
hypothesis = tf.compat.v1.matmul(X, w) + b

loss = tf.compat.v1.reduce_mean(tf.square(y - hypothesis))
opt = tf.compat.v1.train.AdamOptimizer(learning_rate=0.001)
train = opt.minimize(loss=loss)

with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    for step in range(10000):
        sess.run(train, feed_dict={X:X_data, y:y_data})
    pred = sess.run(hypothesis, feed_dict={X:X_data, y:y_data})
print(pred)
