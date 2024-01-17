from sklearn.datasets import load_wine
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder
from keras. callbacks import EarlyStopping,ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler


datasets = load_wine()

X = datasets.data
y = datasets.target
y = y.reshape(-1,1)
# print(X.shape, y.shape) # (178, 13)  (178, )
# print(y)
# print(pd.value_counts(y))       
# 1    71
# 0    59
# 2    48


# 1. scikit-learn 방식
# y = OneHotEncoder(sparse=False).fit_transform(y)
# print(y.shape)
# print(y)            #(178, 3)

# 2. pandas 방식
# y = pd.get_dummies(y)

# 3. keras 방식
y = to_categorical(y)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, shuffle=True, random_state=713, stratify=y)

################    MinMaxScaler    ##############################
# mms = MinMaxScaler()
# mms.fit(X_train)
# X_train = mms.transform(X_train)
# X_test = mms.transform(X_test)

################    StandardScaler    ##############################

# sts = StandardScaler()
# sts.fit(X_train)
# X_train = sts.transform(X_train)
# X_test = sts.transform(X_test)

# print(X_train)
# print(X_test)

# ################    MaxAbsScaler    ##############################
# mas = MaxAbsScaler()
# mas.fit(X_train)
# X_train = mas.transform(X_train)
# X_test = mas.transform(X_test)


# ################    RobustScaler    ##############################
rbs = RobustScaler()
rbs.fit(X_train)
X_train = rbs.transform(X_train)
X_test = rbs.transform(X_test)


model = Sequential()
model.add(Dense(19, input_dim=13,activation='sigmoid'))
model.add(Dense(97))
model.add(Dense(9,activation='relu'))
model.add(Dense(21))
model.add(Dense(3, activation='softmax'))

import datetime
date = datetime.datetime.now()
# print(date)                     # 2024-01-17 10:52:59.857389
date = date.strftime("%m%d_%H%M")                   # _는 str      
# print(date)                     # 0117_1058
# print(type(date))               # <class 'str'>

path = "..\\_data\\_save\\MCP\\"
filename = '{epoch:05d}-{acc:.4f}-{loss:.4f}.hdf5'            # 04d : 4자리 정수표현, 4f : 소수4번째자리까지 표현, 예) 1000_0.3333.hdf5
filepath = "".join([path, 'k26_wine_',date,'_', filename])

mcp = ModelCheckpoint(monitor='val_loss', mode='min', verbose=1, save_best_only=True, filepath=filepath)    
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')
es = EarlyStopping(monitor='val_loss', mode='min', patience=30, restore_best_weights=True, verbose=1)
hist = model.fit(X_train, y_train, epochs=500, batch_size=1, validation_split=0.2, callbacks=[es,mcp], verbose=1)

results = model.evaluate(X_test, y_test)
print("로스 : ", results[0])
print("ACC : ", results[1])

y_predict = model.predict(X_test)
y_test = np.argmax(y_test, axis=1)
y_predict = np.argmax(y_predict, axis=1)

acc = accuracy_score(y_predict, y_test)
print("accuracy_score : ", acc)


# 로스 :  0.2964943051338196
# ACC :  0.8888888955116272
# accuracy_score :  0.8888888888888888





# MinMaxScaler
로스 :  0.0011304776417091489

# MaxAbsScaler
로스 :  0.0020532591734081507
# StandardScaler
로스 :  4.768340886585065e-07

# RobustScaler
로스 :  1.1098047252744436e-05