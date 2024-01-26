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
).next()[0]

print(X_augumented.shape)
X_train = X_train.reshape(-1 , 28 , 28, 1)
X_test = X_test.reshape(-1 , 28 , 28, 1)
print(X_train.shape)

X_train = np.concatenate((X_train, X_augumented))
y_train = np.concatenate((y_train, y_augumented))
print(X_train.shape)
# print(np.unique(y_train, return_counts=True)) # 0~9

####################################
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import OneHotEncoder
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)
ohe = OneHotEncoder(sparse_output=False)
y_train = ohe.fit_transform(y_train)
y_test = ohe.transform(y_test)

model = Sequential()
model.add(Conv2D(30, (3, 3), input_shape=(28,28,1), activation='relu'))
model.add(MaxPooling2D())

model.add(Conv2D(20, (3,3), activation='relu'))
model.add(MaxPooling2D())

# model.add(Conv2D(20, (3,3), activation='relu'))
# model.add(MaxPooling2D())

# model.add(Conv2D(20, (3,3), activation='relu'))
# model.add(MaxPooling2D())

model.add(Flatten())
model.add(Dense(300, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10,activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
es = EarlyStopping(monitor='val_loss', mode='min', patience=30, restore_best_weights=True, verbose=1)
model.fit(X_train, y_train, epochs=1000, batch_size=5000, validation_split=0.2, callbacks=[es])

res = model.evaluate(X_test, y_test)

print("acc : ", res[1])

# 증폭 : 0.9047
# 그냥 : 0.9117