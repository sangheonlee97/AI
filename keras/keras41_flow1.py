import numpy as np

import sys
import tensorflow as tf
import matplotlib.pyplot as plt

print("tensorflow version : ", tf.__version__)
print("python version : ", sys.version)

from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img         # 이미지 땡겨와
from tensorflow.keras.preprocessing.image import img_to_array     # 이미지를 수치화

path = "..//_data//image//catdog//Test//30.jpg"
img = load_img(path, 
                 target_size=(150,150),
                )
# print(img.shape)
print(type(img))
# plt.imshow(img)
# plt.show()
arr = img_to_array(img)
# print(arr)
print(arr.shape)    # (150, 150, 3)
print(type(arr))

# 차원 증가
img = np.expand_dims(arr, axis=0)
print(img.shape)   # (1, 150, 150, 3)

#############################   여기부터 증폭  ####################

datagen = ImageDataGenerator(
    rescale=1/255.,
    # horizontal_flip=True,   # 수평 뒤집기
    # vertical_flip=True,     # 수직 뒤집기
    width_shift_range=0.1,  # 세로로 평행이동
    # height_shift_range=0.1,  # 가로로 평행이동
    # rotation_range=30,       # 정해진 각도만큼 이미지를 회전
    # zoom_range=1.2,         # 축소 또는 확대
    # shear_range=10,        # 좌표 하나를 고정시키고 다른 몇 개의 좌표를 이동시키는 변환
    # fill_mode='nearest',    # 빈 공간을 근사치로 채움,    
)
    
fig, ax = plt.subplots(nrows=3,ncols=5,figsize=(10,10))



it = datagen.flow(img,
                  batch_size=1,
                  )
for i in range(15):
    batch = it.next()
    # print(batch)
    print(batch.shape)
    # image = batch[0].astype('uint8')
    image = batch[0]
    ax[int(i/5)][i%5].imshow(image)
    ax[int(i/5)][i%5].axis('off')
    
    # i   : 0 1 2 3 4 5 6 7 8 9
    # i/5 : 0 0 0 0 0 1 1 1 1 1 
    # i%5 : 0 1 2 3 4 0 1 2 3 4

plt.show()