# X, y 추출해서 모델 만들기. 성능 0.99 이상


from keras.layers import Conv2D, Dense, Flatten, Dropout, MaxPooling2D
import numpy as np
import pandas as pd
from keras.datasets import mnist
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import OneHotEncoder
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

train_datagen = ImageDataGenerator(
    rescale=1/255.,
    # horizontal_flip=True,   # 수평 뒤집기
    # vertical_flip=True,     # 수직 뒤집기
    width_shift_range=0.1,  # 가로로 평행이동
    height_shift_range=0.1, # 세로로 평행이동
    rotation_range=5,       # 정해진 각도만큼 이미지를 회전
    zoom_range=1.2,         # 축소 또는 확대
    shear_range=0.7,        # 좌표 하나를 고정시키고 다른 몇 개의 좌표를 이동시키는 변환
    fill_mode='nearest',    # 빈 공간을 근사치로 채움,    
)

test_datagen = ImageDataGenerator(
    rescale=1./255,   
)

path_train = "..//_data//image//brain//train//"
path_test = "..//_data//image//brain//test//"

Xy_train = train_datagen.flow_from_directory(
    path_train,
    target_size=(200, 200),
    batch_size=160,    
    class_mode='binary',
    shuffle=True,
)

Xy_test = test_datagen.flow_from_directory(
    path_test,
    target_size=(200, 200),
    batch_size=160,    
    class_mode='binary',
    # shuffle=True,
)



(X_train, y_train), (X_test, y_test) = (Xy_train[0][0], Xy_train[0][1]), (Xy_test[0][0], Xy_test[0][1])
print(np.std(X_train), np.mean(X_train))
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


# y_train = y_train.reshape(-1, 1)
# y_test = y_test.reshape( -1, 1)

# ohe = OneHotEncoder(sparse_output=False)
# y_train = ohe.fit_transform(y_train)
# y_test = ohe.transform(y_test)

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
print(np.unique(X_train, return_counts=True))
#2 data
from keras.models import Model, Sequential
from keras.layers import Input
model = Sequential()
model.add(Conv2D(30, (3,3), strides=1, activation='relu', input_shape=(200,200,3)))
model.add(MaxPooling2D())
model.add(Conv2D(40, (3,3), strides=2, activation='relu'))
model.add(MaxPooling2D())
# model.add(Conv2D(40, (3,3), strides=2, activation='relu'))
model.add(Conv2D(40, (3,3), strides=1, activation='relu'))
model.add(MaxPooling2D())

# model.add(Conv2D(40, (3,3), strides=1, activation='relu'))
# model.add(Conv2D(40, (3,3), strides=1, activation='relu'))
model.add(Flatten())
model.add(Dense(500, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(80, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


#3 compile, fit
from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_accuracy', mode='max', patience=30, verbose=1, restore_best_weights=True )
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=5, verbose=1, epochs=10000, validation_split=0.08 , callbacks=[es])

#3 evaluate, predict
res = model.evaluate(X_test, y_test)
print("loss : ", res[0])
print("acc : ", res[1])

