from keras.datasets import imdb
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from keras.utils import pad_sequences
from keras.layers import Dense, Embedding, LSTM
import numpy as np
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=88588)


# print(y_train[:20])
# print(np.unique(y_train, return_counts=True))
# a = []
# for i in X_train:
#     a.append(max(i))
# mxlen = max(a)
# print(mxlen)  # 88587
# print(max(len(i) for i in X_train)) # 2494
avg = sum(len(i) for i in X_train) / 25000
print( avg)

X_train = pad_sequences(X_train, maxlen=200, padding='pre', truncating='pre')
X_test = pad_sequences(X_test, maxlen=200, padding='pre', truncating='pre')
X_add, X_test, y_add, y_test= train_test_split(X_test, y_test, test_size=0.5, stratify=y_test, random_state=42)
X_con = np.concatenate([X_train, X_add],0)
y_con = np.concatenate([y_train, y_add],0)

print(X_con.shape, y_con.shape) # (25000, ) (25000, )
print(X_test.shape, y_test.shape) # (25000, ) (25000, )

model = Sequential()
model.add(Embedding(88587, 100, input_length=200))
model.add(LSTM(50))
model.add(Dense(100, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
es = EarlyStopping(monitor='val_loss', mode='min', patience=70, restore_best_weights=True)
model.fit(X_con, y_con, epochs=1000, batch_size=1000, validation_split=0.2, callbacks=[es])

res = model.evaluate(X_test, y_test)
y_pred = model.predict(X_test)
y_predict = y_pred.round()

f1 = f1_score(y_test, y_predict)
print("acc : ", res[1])
print("f1 : ", f1)