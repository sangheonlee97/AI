from tensorflow.python.keras.layers import Conv2D, Dense, Flatten, Dropout, Reshape, Conv1D, LSTM
import numpy as np
import pandas as pd
from keras.datasets import mnist
from keras.models import Sequential
from sklearn.preprocessing import OneHotEncoder


(X_train, y_train), (X_test, y_test) = mnist.load_data()

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape( -1, 1)

ohe = OneHotEncoder(sparse_output=False)
y_train = ohe.fit_transform(y_train)
y_test = ohe.transform(y_test)

X_train = X_train.reshape(60000, 28, 28, 1).astype('float32')
X_test = X_test.reshape(10000, 28, 28, 1).astype('float32')
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

#2 data
model = Sequential()
model.add(Conv2D(20, (2, 2), input_shape=(28, 28, 1), activation='relu') )      # 27, 27, 20
model.add(Conv2D(30, (4, 4), activation='relu'))                                # 24, 24, 30
model.add(Conv2D(150, (5, 5), activation='relu'))                               # 20, 20, 150
model.add(Dropout(0.3))
model.add(Conv2D(50, (6, 6), activation='relu'))                                # 15, 15, 50
model.add(Reshape(target_shape=(15*15, 50)))                                    # 225, 50
model.add(Conv1D(30, 2))                                                        # 224, 30
model.add(Flatten())                                                            # 6720, 
model.add(Dense(10))                                                            # 10,
model.add(Dense(10))
model.add(Dense(units=64, activation='relu'))
model.add(Dense(10))
model.add(Dense(10, activation='softmax'))
model.summary()
#3 compile, fit
from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', mode='min', patience=30, verbose=1, restore_best_weights=True )
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=500, verbose=1, epochs=5, validation_split=0.2 , callbacks=[es])

#3 evaluate, predict
res = model.evaluate(X_test, y_test)
print("loss : ", res[0])
print("acc : ", res[1])
model.summary()