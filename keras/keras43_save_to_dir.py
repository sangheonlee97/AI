from keras.datasets import fashion_mnist
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
X_train = X_train / 255.
X_test = X_test / 255.
train_datagen = ImageDataGenerator(
    # rescale=1/255. ,
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.2,
    height_shift_range=0.2,
    rotation_range=30,
    zoom_range=0.2,
    fill_mode='nearest',
    shear_range=30
)

augument_size = 40000

randidx = np.random.randint(X_train.shape[0], size=augument_size)
print(randidx.shape)
print(np.unique(randidx,return_counts=True))
print(np.min(randidx), np.max(randidx))

X_augumented = X_train[randidx].copy()      # 40000, 28, 28
y_augumented = y_train[randidx].copy()      # 40000,

X_augumented = train_datagen.flow(
    X_augumented.reshape(-1,28,28,1), y_augumented,
    batch_size=augument_size,
    shuffle=False,
    save_to_dir="..//_data//temp//"
).next()[0]