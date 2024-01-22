from keras.models import Sequential
from keras.layers import Conv2D, Dense, Dropout, Flatten
from keras.datasets import cifar100
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
(X_train, y_train), (X_test, y_test) = cifar100.load_data()

print(X_train.shape, X_test.shape) # (50000, 32, 32, 3) (10000, 32, 32, 3)
# plt.imshow(X_train[2])
# plt.show()
print(y_train.shape, y_test.shape) 
print(np.unique(y_train, return_counts=True)) # 0 ~ 9
print(np.unique(y_test, return_counts=True)) # 0 ~ 9
# print(X_train)
# print(X_test)
print(y_train.shape)
print(y_test.shape)

ohe = OneHotEncoder(sparse=False)
y_train = ohe.fit_transform(y_train)
y_test = ohe.transform(y_test)

####################################
# scaler = MinMaxScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)
# X_train = X_train / 255
# X_test = X_test / 255
####################################

####################################
mean = np.mean(X_train)
std = np.std(X_train)
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std
####################################
X_train = X_train.reshape(50000, -1)
X_test = X_test.reshape(10000, -1)



model = Sequential()
model.add(Dense(5000, input_shape=(3072,),activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7000, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(7000, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(7000, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1000, activation='relu'))
model.add(Dense(100, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
es = EarlyStopping(monitor='val_loss', mode='min', patience=20, restore_best_weights=True, verbose=1)
model.fit(X_train, y_train, epochs=1000, batch_size=500, validation_split=0.15, callbacks=[es], verbose=1)

res = model.evaluate(X_test, y_test)

print("acc : ", res[1])
# 0.2(dnn)