import tensorflow as tf
print(tf.__version__)   # 1.14.0
print(tf.executing_eagerly())   # False
tf.compat.v1.disable_eager_execution()

sess = tf.compat.v1.Session()

a = tf.compat.v1.placeholder(tf.float32)
b = tf.compat.v1.placeholder(tf.float32)
add_node = a + b

print(sess.run(add_node, feed_dict={a:3, b:4}))
print(sess.run(add_node, feed_dict={a:30, b:4.5}))

add_and_triple = add_node * 3


print(sess.run(add_and_triple, feed_dict={a:30, b:4.5}))
