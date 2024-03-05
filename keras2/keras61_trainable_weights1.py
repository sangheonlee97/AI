import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import tensorflow as tf
tf.random.set_seed(777)
np.random.seed(777)
print(tf.__version__)   # 2.9.0

X = np.array([1,2,3,4,5])
y = np.array([1,2,3,4,5])

model = Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(2))
model.add(Dense(1))

model.summary()

# print(model.weights)
for w in model.weights:
    print(w)
    print("\n")
    print("\n")
print("##########################################################################################################################################################################################")
for w in model.trainable_weights:
    print(w)
    print("\n")
    print("\n")
print("##########################################################################################################################################################################################")
    
print(len(model.weights))
print(len(model.trainable_weights))

print("##########################################################################################################################################################################################")


model.trainable = False    # 중요******************

print(len(model.weights))
print(len(model.trainable_weights))
for w in model.weights:
    print(w)
    print("\n")
    print("\n")
print("##########################################################################################################################################################################################")
for w in model.trainable_weights:
    print(w)
    print("\n")
    print("\n")
model.summary()