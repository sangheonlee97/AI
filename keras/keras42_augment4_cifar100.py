from keras.models import Sequential
from keras.layers import Conv2D, Dense, Dropout, Flatten
from keras.datasets import cifar100
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
(X_train, y_train), (X_test, y_test) = cifar100.load_data()
from keras.preprocessing.image import ImageDataGenerator
data_gen = ImageDataGenerator(
    # rescale=1/255. ,
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.2,
    height_shift_range=0.2,
    rotation_range=30,
    zoom_range=0.2,
    fill_mode='nearest',
    shear_range=10
)

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
aug_size = 30000
randidx = np.random.randint(50000, size=aug_size)
X_aug = X_train[randidx].copy()
y_aug = y_train[randidx].copy()
X_aug = data_gen.flow(
    X_aug, y_aug,
    batch_size=aug_size,
    shuffle=False,
).next()[0]

# X_train = np.concatenate((X_train, X_aug), axis=0)
# y_train = np.concatenate((y_train, y_aug), axis=0)




ohe = OneHotEncoder(sparse=False)
y_train = ohe.fit_transform(y_train)
y_test = ohe.transform(y_test)

####################################
# scaler = MinMaxScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)
X_train = X_train / 255.
X_test = X_test / 255.
####################################

####################################
# mean = np.mean(X_train)
# std = np.std(X_train)
# X_train = (X_train - mean) / std
# X_test = (X_test - mean) / std
####################################

from keras.layers import MaxPooling2D

model = Sequential()
model.add(Conv2D(30, (3, 3), input_shape=(32,32,3), activation='relu'))
# model.add(MaxPooling2D())

model.add(Conv2D(20, (3,3), activation='relu'))
model.add(MaxPooling2D())

model.add(Conv2D(20, (3,3), activation='relu'))
# model.add(MaxPooling2D())

model.add(Conv2D(20, (3,3), activation='relu'))
model.add(MaxPooling2D())

model.add(Flatten())
model.add(Dense(300, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(100,activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
es = EarlyStopping(monitor='val_loss', mode='min', patience=20, restore_best_weights=True, verbose=1)
model.fit(X_train, y_train, epochs=1, batch_size=500, validation_split=0.15, callbacks=[es], verbose=1)

res = model.evaluate(X_test, y_test)

print("acc : ", res[1])
import os
print("flie : ",os.path.basename(__file__))
# just  : 0.3962                .3662
# aug   : 0.3862                .3573