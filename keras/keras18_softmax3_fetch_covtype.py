from sklearn.datasets import fetch_covtype
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from keras.utils import to_categorical



datasets = fetch_covtype()
X = datasets.data
y = datasets.target

print(X.shape, y.shape)     # (581012, 54),  (581012, )
print(pd.value_counts(y))
# 2    283301
# 1    211840
# 3     35754
# 7     20510
# 6     17367
# 5      9493
# 4      2747





######### sklearn.preprocessing의 OneHotEncoder###########
# y = y.reshape(-1, 1)
# ohe = OneHotEncoder(sparse=False)
# y = ohe.fit_transform(y)
# print(y.shape)
############################################


######### keras.utils의 to_categorical##########
print(y)
print(pd.value_counts(y-1))
y = to_categorical(y-1)

print(y.shape)      # (581012, 7)

# y = to_categorical(y)
# print(y.shape)      # (581012, 8)
###################
# y = pd.DataFrame(y)
# y = y.drop(y[0], axis=1)
# print(y.shape)      # (581012, 7)
###################


################################################


######### pandas ####################
# y = pd.get_dummies(y)
# print(y.shape)
#####################################






'''
# model = Sequential()
# model.add(Dense(100, input_dim=54))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = Sequential()
model.add(Dense(30, input_dim=54))
model.add(Dense(50))
model.add(Dense(50))
model.add(Dense(30))
model.add(Dense(10))
model.add(Dense(7, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
es = EarlyStopping(monitor='val_loss', mode='min', patience=50, verbose=1, restore_best_weights=True)
model.fit(X_train, y_train, epochs=1000,validation_split=0.2, batch_size=50000, callbacks=[es])

y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)
y_test = np.argmax(y_test, axis=1)
acc = accuracy_score(y_test, y_pred)
print ("acc : ", acc)
'''