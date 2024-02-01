from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences, to_categorical
from keras.models import Sequential
from keras.layers import Dense, LSTM

from sklearn.preprocessing import OneHotEncoder
import numpy as np


# 1
docs = [
    '너무 재미있다',
    '참 최고에요',
    '참 잘만든 영화에요',
    '추천하고 싶은 영화입니다.',
    '한 번 더 보고 싶어요.',
    '글쎄',
    '별로에요',
    '생각보다 지루해요',
    '연기가 어색해요',
    '재미없어요',
    '너무 재미없다.',
    '참 재밌네요.',
    '상헌이 바보',
    '반장 못생겼다',
    '욱이 또 잔다',
]

labels = np.array([1,1,1,1,1,0,0,0,0,0,0,1,2,2,2])
labels = labels.reshape(-1, 1)
ohe = OneHotEncoder(sparse=False)
labels = ohe.fit_transform(labels)
token = Tokenizer()
token.fit_on_texts(docs)
# print(token.word_index)
X = token.texts_to_sequences(docs)
X_padded = pad_sequences(X)
print(X_padded.shape)
import pandas as pd
# X_padded = X_padded.reshape(-1)
X_padded = to_categorical(X_padded)
print(X_padded.shape)
X_padded = X_padded.reshape(15, -1, 31)
print(X_padded.shape)

model = Sequential()
# model.add(LSTM(10, input_shape=(5,1)))
model.add(LSTM(32, input_shape=(5,31)))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_padded, labels, epochs=500, batch_size=1, )
res = model.evaluate(X_padded, labels)
print("loss, acc : ", res)
X_test = [' 처음보는 상헌이 인식 못해? 바보']
token.fit_on_texts(X_test)
X_test = token.texts_to_sequences(X_test)
X_test = pad_sequences(X_test, maxlen=5)
X_test = X_test.reshape(-1, 5)
y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred)

print(y_pred)