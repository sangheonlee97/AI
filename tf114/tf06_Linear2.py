import tensorflow as tf

tf.set_random_seed(123)
sess = tf.compat.v1.Session()

X = [1,2,3,4,5]
y = [1,2,3,4,5]

w = tf.Variable(111, dtype=tf.float32)
b = tf.Variable(0, dtype=tf.float32)

hypothesis = X * w + b


loss = tf.compat.v1.reduce_mean(tf.square(hypothesis - y))
optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
train = optimizer.minimize(loss=loss)

sess.run(tf.global_variables_initializer())

for step in range(1000):
    sess.run(train)
    print(step, sess.run(loss), sess.run(w), sess.run(b))
    
sess.close()
    