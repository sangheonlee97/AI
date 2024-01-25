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
# plt.imshow(X_train[0])
# plt.show()
# print(X_train.shape) # 60000 28 28
X_data = train_datagen.flow(
    np.tile(X_train[0].reshape(28*28),augumet_size).reshape(-1,28,28,1),
    np.zeros(augumet_size),
    batch_size=60,
    shuffle=False
)
fig, ax = plt.subplots(nrows=3,ncols=5,figsize=(10,10))
print("@########################################")
# print(X_data.next()[0].shape)
for i in range(15):
    batch = X_data[1][0][39] 
    # print(batch)
    # print(batch.shape)
    # image = batch[0].astype('uint8')
    image = batch
    ax[int(i/5)][i%5].imshow(image)
    ax[int(i/5)][i%5].axis('off')
    
    # i   : 0 1 2 3 4 5 6 7 8 9
    # i/5 : 0 0 0 0 0 1 1 1 1 1 
    # i%5 : 0 1 2 3 4 0 1 2 3 4

plt.show()