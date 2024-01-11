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

# print(X.shape, y.shape)     # (150, 4) (150,)
# print(np.unique(y, return_counts=True))
# print(pd.value_counts(y))

######### sklearn.preprocessing의 OneHotEncoder###########
y = y.reshape(-1, 1)
ohe = OneHotEncoder(sparse=False)
print(type(ohe.fit(y)))
y = ohe.transform(y)
############################################


######### keras.utils의 to_categorical##########
#y_ohe = to_categorical(y)
################################################


######### pandas ####################
#y = pd.get_dummies(y)
#####################################


'''
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = Sequential()
model.add(Dense(20, input_dim=4))
model.add(Dense(30))
model.add(Dense(30))
model.add(Dense(30))
model.add(Dense(30))
model.add(Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
es = EarlyStopping(monitor='val_loss', patience=130, mode='min', verbose=1, restore_best_weights=True)
model.fit(X_train, y_train, epochs=10000, batch_size=5, validation_split=0.3, callbacks=[es])


y_pred = model.predict(X_test)
model.evaluate(X_test)
# acc = accuracy_score(y_test, y_pred)

# print('acc : ', acc)
'''