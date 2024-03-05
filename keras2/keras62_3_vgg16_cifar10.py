import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.datasets import cifar10
import tensorflow as tf
tf.random.set_seed(777)
np.random.seed(777)
print(tf.__version__)   # 2.9.0

from keras.applications import VGG16
from keras.callbacks import EarlyStopping
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train = X_train / 255.
X_test = X_test / 255.
vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
vgg16.trainable = False     # 가중치 동결
model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(100, activation='swish'))
model.add(Dense(100, activation='swish'))
model.add(Dense(100, activation='swish'))
model.add(Dense(10, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(monitor='val_loss', mode='min', patience=30, restore_best_weights=True)
model.fit(X_train, y_train, validation_split=0.2, batch_size=500, epochs=100, callbacks=[es])

ree = model.evaluate(X_test, y_test)
print("acc : ", ree[1])
# vgg16 / trainable : False
# acc :  0.6032000184059143

# cnn
# acc :  0.6866000294685364