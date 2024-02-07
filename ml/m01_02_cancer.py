import numpy as np
from sklearn.datasets import load_breast_cancer
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, accuracy_score
import pandas as pd
from sklearn.svm import LinearSVC

#1. data
datasets = load_breast_cancer()
# print(datasets.DESCR)
# print(datasets.feature_names)

X = datasets.data
y = datasets.target        # (569, 30)
print(X.shape, y.shape)     # (569, )
######################################
# print(np.unique(y)) # [0 1] ## np.unique(y, return_counts=True)
# pd_y = pd.DataFrame(y)
# print(pd_y)
# print("0 : ", pd_y[pd_y == 0].count())
# print("1 : ", pd_y[pd_y == 1].count())
######################################
a1 = np.where(y==0)
a2 = np.where(y==1)
print("0 : ", len(a1[0]) )
print("1 : ", len(a2[0]) )

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#2. model

model = LinearSVC()

model.fit(X_train, y_train)

#4. evaluate, pred
y_pred = model.predict(X_test)

y_pred = y_pred.round()

loss = model.score(X_test, y_test)
r2 = r2_score(y_test, y_pred)

def ACC(a, b):
    return accuracy_score(a,b)
acc = ACC(y_test, y_pred)
print(y_pred)  
print("loss : ", loss)
print("r2 : ", r2)
print("acc : ", acc)















''' 회귀 모델
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)


#2. model
model = Sequential()
model.add(Dense(23, input_dim=30, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(1))

#3. compile, fit
model.compile(loss='mse', optimizer='adam')
es = EarlyStopping(monitor='val_loss', mode='min', patience=50, verbose=1)
model.fit(X_train, y_train, epochs=1000, validation_split=0.1, callbacks=[es])

#4. evaluate, pred
y_pred = model.predict(X_test)

idx=0
for i in y_pred:
    if 3-i > 2.5:
        y_pred[idx] = 0
    else:
        y_pred[idx] = 1
    idx += 1

print(y_pred)

loss = model.evaluate(X_test, y_test)
r2 = r2_score(y_test, y_pred)

        
print("loss : ", loss)
print("r2 : ", r2)

# 2/2 [==============================] - 0s 0s/step - loss: 0.0291
# loss :  0.029143579304218292
# r2 :  1.0


# [[1.]
#  [0.]
#  [0.]
#  [1.]
#  [1.]
#  [0.]
#  [0.]
#  [0.]
#  [1.]
#  [1.]
#  [1.]
#  [0.]
#  [1.]
#  [0.]
#  [1.]
#  [0.]
#  [1.]
#  [1.]
#  [1.]
#  [0.]
#  [0.]
#  [1.]
#  [0.]
#  [1.]
#  [1.]
#  [1.]
#  [1.]
#  [1.]
#  [1.]
#  [0.]
#  [1.]
#  [1.]
#  [1.]
#  [1.]
#  [1.]
#  [1.]
#  [0.]
#  [1.]
#  [0.]
#  [1.]
#  [1.]
#  [0.]
#  [1.]
#  [1.]
#  [1.]
#  [1.]
#  [1.]
#  [1.]
#  [1.]
#  [1.]
#  [0.]
#  [0.]
#  [1.]
#  [1.]
#  [1.]
#  [1.]
#  [1.]]'''