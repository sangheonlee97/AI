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




#1. data
datasets = load_iris()
# print(datasets)
# print(datasets.DESCR)
# print(datasets.feature_names)
X = datasets.data
y = datasets.target

print(datasets.DESCR)

# print(X.shape, y.shape)     # (150, 4) (150,)
# print(np.unique(y, return_counts=True))
# print(pd.value_counts(y))

######### sklearn.preprocessing의 OneHotEncoder###########
y = y.reshape(-1, 1)
ohe = OneHotEncoder(sparse=False)
# ohe.fit(y)
# y = ohe.transform(y)
y = ohe.fit_transform(y)
############################################


######### keras.utils의 to_categorical##########
#y_ohe = to_categorical(y)
################################################


######### pandas ####################
#y = pd.get_dummies(y)
#####################################



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y )

model = Sequential()
model.add(Dense(20, input_dim=4))
model.add(Dense(30))
model.add(Dense(30))
model.add(Dense(30))
model.add(Dense(30))
model.add(Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
es = EarlyStopping(monitor='val_loss', patience=130, mode='min', verbose=1, restore_best_weights=True)
model.fit(X_train, y_train, epochs=10000, batch_size=5, validation_split=0.2, callbacks=[es])


y_pred = model.predict(X_test)
results = model.evaluate(X_test, y_test)


# print(y_pred.shape)     # (45, 3)


print('loss : ', results[0])
# print('acc : ', results[1])
y_test = np.argmax(y_test, axis=1)
y_pred = np.argmax(y_pred, axis=1)
acc = accuracy_score(y_test, y_pred)
print("acc : ", acc)