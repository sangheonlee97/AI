import tensorflow as tf
import matplotlib.pyplot as plt
tf.set_random_seed(777)

# 1. data
X_train = [1, 2, 3]
y_train = [1, 2, 3]
X = tf.compat.v1.placeholder(tf.float32)
y = tf.compat.v1.placeholder(tf.float32)

w = tf.compat.v1.Variable([10], dtype=tf.float32, name='weights')

# 2. model
hypothesis = X * w

# 3. compile    // model.compile(loss='mse', optimizer='sgd')
loss = tf.reduce_mean(tf.square(hypothesis - y))

############## optimizer ##############
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
# train = optimizer.minimize(loss)
lr = 0.01
gradient = tf.reduce_mean((X * w - y) * X)
descent = w - lr * gradient
update = w.assign(descent)
############## optimizer ##############

sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())

w_history = []
loss_history = []


for step in range(100):
    _, loss_v, w_v = sess.run([update, loss, w], feed_dict={X:X_train, y:y_train})
    print(step, '\t', loss_v, '\t', w_v)
    w_history.append(w_v)
    loss_history.append(loss_v)
sess.close()

plt.plot(loss_history)
plt.xlabel('epochs')
plt.ylabel('loss')
plt.show()
