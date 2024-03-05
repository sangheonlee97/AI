import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import tensorflow as tf
tf.random.set_seed(777)
np.random.seed(777)
print(tf.__version__)   # 2.9.0

from keras.applications import VGG16

model = VGG16( include_top=False)#, input_shape=(100,100,3))
model.summary()

################### include_top = False #####################
#1. FC layer 날려
#2. input_shape 내가 하고 싶은걸로 해
