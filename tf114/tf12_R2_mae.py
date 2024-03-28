import tensorflow as tf
import matplotlib.pyplot as plt
tf.compat.v1.set_random_seed(777)

# 1. data
X_train = [1, 2, 3]
y_train = [1, 2, 3]
X_test = [4, 5, 6]
y_test = [4, 5, 6]
X = tf.compat.v1.placeholder(tf.float32)
y = tf.compat.v1.placeholder(tf.float32)

w = tf.compat.v1.Variable([10], dtype=tf.float32, name='weights')
# b = tf.compat.v1.Variable([1], dtype=tf.float32, name='weights')

# 2. model
# hypothesis = X * w + b
hypothesis = X * w

# 3. compile    // model.compile(loss='mse', optimizer='sgd')
loss = tf.reduce_mean(tf.square(hypothesis - y))

############## optimizer ##############
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
# train = optimizer.minimize(loss)
lr = 0.01
# gradient = tf.reduce_mean((X * w + b - y) * X)
gradient = tf.reduce_mean((X * w - y) * X)
descent = w - lr * gradient
update = w.assign(descent)
############## optimizer ##############

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

w_history = []
loss_history = []


for step in range(200):
    _, loss_v, w_v = sess.run([update, loss, w], feed_dict={X:X_train, y:y_train})
    print(step, '\t', loss_v, '\t', w_v[0])
    w_history.append(w_v)
    loss_history.append(loss_v)
pred = sess.run(hypothesis, feed_dict={X: X_test, y:y_test})
print('y_pred : ', pred)
sess.close()

# plt.plot(loss_history)
# plt.xlabel('epochs')
# plt.ylabel('loss')
# plt.show()
#### 실습 R2, mae 맹그러 ####
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
r2 = r2_score(y_test, pred)
mae = mean_absolute_error(y_test, pred)
mse = mean_squared_error(y_test, pred)
print('r2 : ', r2)
print('mae : ', mae)
print('mse : ', mse)