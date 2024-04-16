from keras.datasets import cifar100
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPooling2D, GlobalAveragePooling2D
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np
(X_train, y_train), (X_test, y_test) = cifar100.load_data()

print(X_train.shape)
print(X_test.shape)
# print(pd.value_counts(y_train))
# plt.imshow(X_train[5])
# plt.show()

X_train = X_train.reshape(-1, 32, 32, 3)
X_test = X_test.reshape(-1, 32, 32, 3)

print(y_train.shape)
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

ohe = OneHotEncoder(sparse=False)
y_train = ohe.fit_transform(y_train)
y_test = ohe.transform(y_test)

print(X_train.shape)
print(X_test.shape)
model = Sequential()
model.add(Conv2D(150, (3,3), padding='same', strides=2, input_shape=(32,32,3), activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(180, (3,3), padding='valid', activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(180, (3,3), padding='same', activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(180, (3,3), padding='same', activation='relu'))
model.add(Conv2D(180, (3,3), padding='same', activation='relu'))
model.add(GlobalAveragePooling2D())
model.add(Dense(500))
model.add(Dropout(0.5))
model.add(Dense(500))
model.add(Dropout(0.5))
model.add(Dense(100, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
es = EarlyStopping(monitor='val_loss', mode='min', patience=50, verbose=1, restore_best_weights=True)
model.fit(X_train, y_train, epochs=300, batch_size=500, callbacks=[es], validation_split=0.2)

res = model.evaluate(X_test, y_test)

print("acc : ", res[1])

# Flatten : 0.30730000138282776

# GlobalAveragePooling2D : 0.30399999022483826