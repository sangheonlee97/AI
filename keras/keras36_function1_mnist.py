from keras.layers import Conv2D, Dense, Flatten, Dropout
import numpy as np
import pandas as pd
from keras.datasets import mnist
from keras.models import Sequential
from sklearn.preprocessing import OneHotEncoder


(X_train, y_train), (X_test, y_test) = mnist.load_data()

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape( -1, 1)

ohe = OneHotEncoder(sparse_output=False)
y_train = ohe.fit_transform(y_train)
y_test = ohe.transform(y_test)

# print(X_train[0])       # 5
# print(pd.value_counts(y_train))
# print(np.unique(y_train, return_counts=True))

X_train = X_train.reshape(60000, 28, 28, 1).astype('float32')
# X_test = X_test.reshape(10000, 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1).astype('float32')
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
print(np.unique(X_train, return_counts=True))
#2 data
# model = Sequential()
# model.add(Conv2D(20, (2, 2), input_shape=(28, 28, 1), activation='relu') )
# model.add(Conv2D(30, (4, 4), activation='relu'))
# model.add(Conv2D(150, (5, 5), activation='relu'))
# model.add(Dropout(0.3))
# model.add(Conv2D(50, (6, 6), activation='relu'))
# model.add(Conv2D(60, (7, 7), activation='relu'))
# model.add(Conv2D(30, (8, 8), activation='relu'))
# model.add(Flatten())
# model.add(Dense(10))
# model.add(Dense(10))
# model.add(Dense(10))
# model.add(Dense(10, activation='softmax'))
from keras.models import Model
from keras.layers import Input
ip = Input(shape=(28,28,1))
d1 = Conv2D(20, (2, 2), activation='relu')(ip)
d2 = Conv2D(30, (4, 4), activation='relu')(d1)
d3 = Conv2D(150, (5, 5), activation='relu')(d2)
dr1 = Dropout(0.3)(d3)
d4 = Conv2D(50, (6, 6), activation='relu')(dr1)
d5 = Conv2D(60, (7, 7), activation='relu')(d4)
d6 = Conv2D(30, (8, 8), activation='relu')(d5)
f = Flatten()(d6)
d7 = Dense(10)(f)
d8 = Dense(10)(d7)
d9 = Dense(10)(d8)
op = Dense(10, activation='softmax')(d9)
model = Model(inputs=ip, outputs=op)

#3 compile, fit
from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', mode='min', patience=30, verbose=1, restore_best_weights=True )
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=500, verbose=1, epochs=10000, validation_split=0.2 , callbacks=[es])

#3 evaluate, predict
res = model.evaluate(X_test, y_test)
print("loss : ", res[0])
print("acc : ", res[1])

