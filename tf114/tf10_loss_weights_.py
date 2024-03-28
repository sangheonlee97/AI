import tensorflow as tf
import matplotlib.pyplot as plt
tf.set_random_seed(777)

X = [1, 2, 3]
y = [1, 2, 3]

w = tf.compat.v1.placeholder(tf.float32)

hypothesis = X * w

loss = tf.reduce_mean(tf.square(hypothesis - y))

w_history = []
loss_history = []

with tf.compat.v1.Session() as sess:
    for i in range(-30, 50):
        curr_w = i
        curr_loss = sess.run(loss, feed_dict={w:curr_w})
        
        w_history.append(curr_w)
        loss_history.append(curr_loss)

plt.plot(w_history, loss_history)
plt.xlabel('w')
plt.ylabel('loss')
plt.show()
