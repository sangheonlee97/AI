import tensorflow as tf

tf.set_random_seed(123)
# sess = tf.compat.v1.Session()
with tf.compat.v1.Session() as sess:
    X = [1,2,3]
    y = [1,2,3]

    w = tf.Variable(111, dtype=tf.float32)
    b = tf.Variable(0, dtype=tf.float32)

    # hypothesis = w * X + b      # 이거 아니야. 이제는 말할 수 있다!!
    hypothesis = X * w + b

    loss = tf.compat.v1.reduce_mean(tf.square(hypothesis - y))      # mse
    optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
    train = optimizer.minimize(loss=loss)

    sess.run(tf.global_variables_initializer())
    epochs = 100
    for step in range(epochs):
        sess.run(train)
        print(step, sess.run(loss), sess.run(w), sess.run(b))

    # sess.close()