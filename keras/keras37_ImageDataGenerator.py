from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train_datagen = ImageDataGenerator(
    rescale=1/255.,
    horizontal_flip=True,   # 수평 뒤집기
    vertical_flip=True,     # 수직 뒤집기
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

# for i in Xy_train:
#     print(i)
print(Xy_train[0][0].shape)     # batch_size, target_size( , ), 3

print(type(Xy_train[0][0]))