from keras.utils import pad_sequences
from sklearn.preprocessing import OneHotEncoder
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Embedding, LSTM, Dropout, Conv1D, Flatten
from keras.datasets import reuters
from statistics import median
import numpy as np
import pandas as pd
(X_train, y_train), (X_test, y_test) = reuters.load_data(num_words=2451532456835890000, test_split=0.2, )
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

print(len(np.unique(y_train)))  # 46

print(len(X_train[0]))  # 87
print(len(X_train[1]))  # 56
print("최대 길이 : ", max(len(i) for i in X_train))             # 2376
print("평균 길이 : ", sum(map(len, X_train)) / len(X_train))    # 145
print("median : ", median(len(i) for i in X_train))             # 95
print("max : ", max(max(i) for i in X_train))                   # 30982
# print(np.unique(X_train))  # 46



X_train = pad_sequences(X_train, padding='pre', maxlen=70, truncating='pre')
X_test = pad_sequences(X_test, padding='pre', maxlen=70, truncating='pre')

# X_train = X_train.reshape(-1, 70, 1)
# X_test = X_test.reshape(-1, 70, 1)

# y 원핫 할람하고 하기 싫으면 sparse_categorical_crossentropy 쓰면됨
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)
ohe = OneHotEncoder(sparse_output=False)
y_train = ohe.fit_transform(y_train)
y_test = ohe.transform(y_test)

model = Sequential()
model.add(Embedding(input_dim=30983, output_dim=200, input_length=70))
# model.add(LSTM(64))
model.add(Conv1D(64, 3))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(46, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
es = EarlyStopping(monitor='val_loss', mode='min', patience=100, restore_best_weights=True)
model.fit(X_train, y_train, epochs=1000, batch_size=300, validation_split=0.2, callbacks=[es])

res = model.evaluate(X_test, y_test)

print("loss : ", res[0])
print("acc : ", res[1])