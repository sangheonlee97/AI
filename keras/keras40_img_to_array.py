import numpy as np

import sys
import tensorflow as tf
import matplotlib.pyplot as plt

print("tensorflow version : ", tf.__version__)
print("python version : ", sys.version)

from keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.preprocessing.image import load_img         # 이미지 땡겨와
# from tensorflow.keras.preprocessing.image import img_to_array     # 이미지를 수치화
from keras.utils import load_img, img_to_array

path = "..//_data//image//catdog//Test//30.jpg"
img = load_img(path, 
                 target_size=(150,150),
                )
# plt.imshow(img)
# plt.show()
arr = img_to_array(img)
print(arr)
print(arr.shape)    # (150, 150, 3)
print(type(arr))

# 차원 증가
img = np.expand_dims(arr, axis=0)
print(img.shape)   # (1, 150, 150, 3)