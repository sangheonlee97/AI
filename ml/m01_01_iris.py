import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from keras.utils import to_categorical
from sklearn.svm import LinearSVC



datasets = load_iris()
X = datasets.data
y = datasets.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y )

# model = Sequential()
# model.add(Dense(20, input_dim=4))
# model.add(Dense(30))
# model.add(Dense(30))
# model.add(Dense(30))
# model.add(Dense(30))
# model.add(Dense(3, activation='softmax'))
model = LinearSVC(C=110)

# model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# es = EarlyStopping(monitor='val_loss', patience=130, mode='min', verbose=1, restore_best_weights=True)
# model.fit(X_train, y_train, epochs=10, batch_size=5, validation_split=0.2)
model.fit(X_train,y_train)

results = model.score(X_test, y_test)

print('score : ', results)
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print("acc : ", acc)