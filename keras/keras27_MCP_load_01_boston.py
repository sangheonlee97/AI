# keras09_1_boston.py

# MinMaxScaler : 0에서 1사이 X 데이터값 조정

import warnings
warnings.filterwarnings('ignore')

from sklearn.datasets import load_boston
from keras.models import Sequential,load_model
from keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split
import time
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler
from keras.callbacks import EarlyStopping, ModelCheckpoint

datasets = load_boston()
# print(datasets)

X = datasets.data
y = datasets.target




X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, shuffle=True, random_state=20)

################    MinMaxScaler    ##############################
mms = MinMaxScaler()
mms.fit(X_train)
X_train = mms.transform(X_train)
X_test = mms.transform(X_test)

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
# rbs = RobustScaler()
# rbs.fit(X_train)
# X_train = rbs.transform(X_train)
# X_test = rbs.transform(X_test)



# print(np.min(X_train))      # 0.0
# print(np.min(X_test))       # 0.0
# print(np.max(X_train))      # 1.0
# print(np.max(X_test))       # 1.210017220702162

# model = Sequential()
# model.add(Dense(19,input_dim=13,activation='relu'))
# model.add(Dense(97,activation='relu'))
# model.add(Dense(9,activation='relu'))
# model.add(Dense(21,activation='relu'))
# model.add(Dense(28,activation='relu'))
# model.add(Dense(1))

# mcp = ModelCheckpoint(monitor='val_loss', mode='min', verbose=1, save_best_only=True, filepath="c:\\_data\\_save\\MCP\\keras26_MCP01_boston.hdf5")    
# model.compile(loss='mse', optimizer='adam')                                                             # early stopping 개념, min,max, auto
# es = EarlyStopping(monitor='val_loss', mode='min',patience=100, verbose= 1, restore_best_weights=True) 
# # start_time = time.time()
# model.fit(X_train, y_train, epochs=1500, batch_size=15, validation_split=0.1, callbacks=[es,mcp] )
# # end_time = time.time()

model = load_model("..\\_data\\_save\\MCP\\k26_1_boston_0117_1308_00100-11.1892-8.4118.hdf5")

loss = model.evaluate(X_test, y_test)
print("로스 : ", loss)

y_predict = model.predict(X_test)
result = model.predict(X)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("R2스코어 : ", r2)
