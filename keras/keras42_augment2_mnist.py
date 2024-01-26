from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPooling2D
from sklearn.preprocessing import OneHotEncoder
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

(X_train, y_train), (X_test, y_test) = mnist.load_data()

print(X_train.shape, y_train.shape) # 60000 28 28
print(X_test.shape, y_test.shape)   # 10000 28 28
aug_size = 40000
randidx = np.random.randint(X_train.shape[0],size=aug_size)

X_train = X_train.reshape(-1, 28,28,1)
y_train = y_train.reshape(-1, 1)
X_test = X_test.reshape(-1, 28, 28,1 )
X_aug = X_train[randidx].copy()
y_aug = y_train[randidx].copy()

data_gen = ImageDataGenerator(
    rescale=1/255. ,
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.2,
    height_shift_range=0.2,
    rotation_range=30,
    zoom_range=0.2,
    fill_mode='nearest',
    shear_range=10
)

print("sadlfkasdj;sdafl;", X_aug.shape)

X_aug = data_gen.flow(
                    X_aug,
                    y_aug,
                    batch_size=aug_size,
                    shuffle=False
                 ).next()[0]
X_train = np.concatenate((X_train, X_aug),axis=0)
y_train = np.concatenate((y_train, y_aug),axis=0)
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

# 그냥 : 0.9832
# 증폭 : 0.9886