from keras.datasets import fashion_mnist
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
train_datagen = ImageDataGenerator(
    rescale=1/255. ,
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.2,
    height_shift_range=0.2,
    rotation_range=30,
    zoom_range=0.2,
    fill_mode='nearest',
    shear_range=30
    
)

augumet_size = 100
# plt.imshow(X_train[0])
# plt.show()
# print(X_train.shape) # 60000 28 28
X_data = train_datagen.flow(  # tuple로 return
    np.tile(X_train[0].reshape(-1),augumet_size).reshape(-1,28,28,1),
    np.zeros(augumet_size),
    batch_size=100,
    shuffle=False
).next()
# fig, ax = plt.subplots(nrows=3,ncols=5,figsize=(10,10))
print("@########################################")
# print(X_data.next()[0].shape)
plt.figure(figsize=(7,7))

for i in range(49):
    plt.subplot(7,7,i+1)
    plt.axis('off')
    plt.imshow(X_data[0][i], cmap='gray')
    
    # i   : 0 1 2 3 4 5 6 7 8 9
    # i/5 : 0 0 0 0 0 1 1 1 1 1 
    # i%5 : 0 1 2 3 4 0 1 2 3 4

plt.show()