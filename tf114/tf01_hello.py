import tensorflow as tf
print(tf.__version__)

print("hello world")

hello = tf.constant('tensorflow hello world')
print(hello)

session = tf.Session()
print(session.run(hello))