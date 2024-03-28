import tensorflow as tf
tf.compat.v1.set_random_seed(42)

X1_data = [73., 93., 89., 96., 73.]
X2_data = [80., 88., 91., 98., 66.]
X3_data = [75., 93., 90., 100., 70.]
y_data = [152., 185., 180., 196., 142.]
lr = 0.001
epoch = 10000
# 맹그러

X1 = tf.compat.v1.placeholder(tf.float32)
X2 = tf.compat.v1.placeholder(tf.float32)
X3 = tf.compat.v1.placeholder(tf.float32)
y = tf.compat.v1.placeholder(tf.float32)

w1 = tf.compat.v1.Variable([1], dtype=tf.float32)
w2 = tf.compat.v1.Variable([1], dtype=tf.float32)
w3 = tf.compat.v1.Variable([1], dtype=tf.float32)
b = tf.compat.v1.Variable([1], dtype=tf.float32)

hypothesis = X1*w1 + X2*w2 + X3*w3 + b
loss = tf.reduce_mean(tf.square(y - hypothesis))
opt = tf.train.AdamOptimizer(learning_rate=lr)
train = opt.minimize(loss=loss)

with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    for step in range(epoch):
        sess.run(train,feed_dict={X1:X1_data, X2:X2_data, X3:X3_data, y:y_data})
    predict = sess.run(hypothesis, feed_dict={X1:X1_data, X2:X2_data, X3:X3_data, y:y_data})
print('pred : ', predict)