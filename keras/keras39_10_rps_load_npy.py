import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping

np_path = "..\\_data\\_save_npy\\rps\\"

X = np.load(np_path + "rps_X.npy")
y = np.load(np_path + "rps_y.npy")
test_X = np.load(np_path + "hand_X.npy")
test_y = np.load(np_path + "hand_y.npy")
print(np.unique(y, return_counts=True))
print(y)
# ohe = OneHotEncoder(sparse=False)
# y = ohe.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

model = Sequential()
model.add(Conv2D(30, (3, 3), input_shape=(150,150,3), activation='relu'))
model.add(MaxPooling2D())

model.add(Conv2D(20, (3,3), activation='relu'))
model.add(MaxPooling2D())

model.add(Conv2D(20, (3,3), activation='relu'))
model.add(MaxPooling2D())

model.add(Conv2D(20, (3,3), activation='relu'))
model.add(MaxPooling2D())

model.add(Flatten())
model.add(Dense(300, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(3,activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
es = EarlyStopping(monitor='val_loss', mode='min', patience=10, restore_best_weights=True, verbose=1)
model.fit(X_train, y_train, epochs=1000, batch_size=5, validation_split=0.2, callbacks=[es])

res = model.evaluate(X_test, y_test)

print("acc : ", res[1])

res_real = model.evaluate(test_X, test_y)


print("내 손 : ", res_real[1])
