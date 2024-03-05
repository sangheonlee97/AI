from keras.models import Sequential
from keras.layers import Conv2D, Dense, Dropout, Flatten
from keras.datasets import cifar10
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import numpy as np
import pandas as pd
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

print(X_train.shape, X_test.shape) # (50000, 32, 32, 3) (10000, 32, 32, 3)
# print(y_train.shape, y_test.shape) 
# print(np.unique(y_train, return_counts=True)) # 0 ~ 9
# print(np.unique(y_test, return_counts=True)) # 0 ~ 9
# print(X_train)
# print(X_test)
# print(y_train)
# print(y_test)

ohe = OneHotEncoder(sparse=False)
y_train = ohe.fit_transform(y_train)
y_test = ohe.transform(y_test)

# scaler = MinMaxScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)
X_train = X_train / 255
X_test = X_test / 255

model = Sequential()
model.add(Conv2D(120, (3, 3), input_shape=(32, 32, 3), activation='relu') )
model.add(Conv2D(130, (4, 4), activation='relu'))
model.add(Conv2D(130, (4, 4), activation='relu'))
model.add(Conv2D(150, (3, 3), activation='relu'))
model.add(Conv2D(160, (3, 3), activation='relu'))
model.add(Conv2D(130, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
es = EarlyStopping(monitor='val_loss', mode='min', patience=30, restore_best_weights=True)
model.fit(X_train, y_train, validation_split=0.2, batch_size=500, epochs=100, callbacks=[es])

res = model.evaluate(X_test, y_test)

print("acc : ", res[1])
