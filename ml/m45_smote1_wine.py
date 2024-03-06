import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping

# 1. data
dataset = load_wine()
X = dataset.data
y = dataset.target

X = X[:-30]
y = y[:-30]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# 2. model
model = Sequential()
model.add(Dense(10, input_shape=(13,),activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(3, activation='softmax'))

# 3. compile
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
es = EarlyStopping(monitor='val_loss', mode='min', patience=30, restore_best_weights=True)
model.fit(X_train, y_train, epochs=100, batch_size=16, validation_split=0.2, callbacks=[es])

# 4. predict
res = model.evaluate(X_test, y_test)

print("acc : ", res[1])
y_pre  = model.predict(X_test)
y_pre = np.argmax(y_pre, axis=1)
acc = accuracy_score(y_test, y_pre)
print(acc)
f1 = f1_score(y_test, y_pre, average='macro')
print("f1 : " , f1)