from sklearn.datasets import load_wine
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from keras.utils import to_categorical

datasets = load_wine()
X = datasets.data
y = datasets.target

print(X.shape, y.shape)     # (178, 13), (178, )
print(pd.value_counts(y))
# 1    71
# 0    59
# 2    48


######### sklearn.preprocessing의 OneHotEncoder###########
# y = y.reshape(-1, 1)
# ohe = OneHotEncoder(sparse=False)
# # ohe.fit(y)
# # y = ohe.transform(y)
# y = ohe.fit_transform(y)
############################################


######### keras.utils의 to_categorical##########
# y = to_categorical(y)
################################################


######### pandas ####################
y = pd.get_dummies(y)
#####################################

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = Sequential()
model.add(Dense(30, input_dim=13))
model.add(Dense(50))
model.add(Dense(50))
model.add(Dense(30))
model.add(Dense(10))
model.add(Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
es = EarlyStopping(monitor='val_loss', mode='min', patience=50, verbose=1, restore_best_weights=True)
model.fit(X_train, y_train, epochs=1000,validation_split=0.2, batch_size=10, callbacks=[es])

y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)
y_test = np.argmax(y_test, axis=1)
acc = accuracy_score(y_test, y_pred)
print ("acc : ", acc)