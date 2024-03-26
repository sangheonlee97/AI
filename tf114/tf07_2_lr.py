import tensorflow as tf
import random
tf.set_random_seed(777)
sess = tf.compat.v1.Session()

X_data = [1, 2, 3, 4, 5]
y_data = [3, 5, 7, 9, 11]
X = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None])
y = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None])

w = tf.compat.v1.Variable(tf.random_normal([1]), dtype=tf.float32)
b = tf.compat.v1.Variable(tf.random_normal([1]), dtype=tf.float32)


hypothesis = X * w + b

loss = tf.compat.v1.reduce_mean(tf.square(hypothesis - y))
while 1:
    r = random.uniform(0.00001, 0.1)
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=r)
    train = optimizer.minimize(loss=loss)

    sess.run(tf.compat.v1.global_variables_initializer())

    for step in range(101):
        _, loss_val, w_val, b_val = sess.run([train, loss, w, b], feed_dict={X : X_data , y : y_data })
        print("%-5s" % step, '\t',  "%-12s" % loss_val,'\t',  "%-10s" % w_val[0],'\t', "%-12s" %  b_val[0])
    if abs(w_val - 2.0) <= 0.01:
        if abs(b_val - 1.0) <= 0.01:
            break
        

X_pred = [6, 7, 8]
X_test = tf.compat.v1.placeholder(tf.float32, shape=[None])
y_pred = X_test * w_val + b_val
predict = sess.run(y_pred, feed_dict={X_test : X_pred})
print('[6, 7, 8]의 예측값 : [{}, {}, {}]'.format(predict[0], predict[1], predict[2]))
print('lr : ', r)

sess.close()
# [6, 7, 8]의 예측값 : [12.995850563049316, 14.99455738067627, 16.993263244628906]
# lr :  0.009729748650009558