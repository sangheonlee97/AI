from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, Flatten
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

print(X_train.shape)
print(X_test.shape)
# print(pd.value_counts(y_train))
# plt.imshow(X_train[5])
# plt.show()

X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

print(y_train.shape)
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

ohe = OneHotEncoder(sparse=False)
y_train = ohe.fit_transform(y_train)
y_test = ohe.transform(y_test)

print(X_train.shape)
print(X_test.shape)
model = Sequential()
model.add(Conv2D(150, (3,3), input_shape=(28,28,1), activation='relu'))
model.add(Conv2D(180, (3,3), activation='relu'))
model.add(Dropout(0.3))
model.add(Conv2D(180, (3,3), activation='relu'))
model.add(Dropout(0.3))
model.add(Conv2D(180, (3,3), activation='relu'))
model.add(Dropout(0.3))
model.add(Conv2D(180, (3,3), activation='relu'))
model.add(Dropout(0.3))
model.add(Conv2D(80, (3,3), activation='relu'))
model.add(Flatten())
model.add(Dense(500))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))



model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
es = EarlyStopping(monitor='val_loss', mode='min', patience=20, verbose=1, restore_best_weights=True)
model.fit(X_train, y_train, epochs=300, batch_size=500, callbacks=[es], validation_split=0.2)

res = model.evaluate(X_test, y_test)

print("acc : ", res[1])
