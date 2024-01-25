from keras.datasets import fashion_mnist
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
train_datagen = ImageDataGenerator(
    rescale=1/255. ,
    horizontal_flip=True,
    fill_mode='nearest',
    
)

augumet_size = 100
plt.imshow(X_train[0])
plt.show()
print(X_train.shape) # 60000 28 28