from sklearn.datasets import load_diabetes
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import time
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler

datasets = load_diabetes()
X = datasets.data
y = datasets.target          
print(X.shape)  # 442 10
X = X.reshape(442,5,2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=True, random_state=1226)           # 1226 713

# mms = MinMaxScaler()
# mms.fit(X_train)
# X_train = mms.transform(X_train)
# X_test = mms.transform(X_test)


# model = Sequential()
# model.add(Dense(8,input_dim=10))
# model.add(Dense(16))
# model.add(Dense(24))
# model.add(Dense(16))
# model.add(Dense(8))
# model.add(Dense(1))

from keras.models import Model
from keras.layers import Input, LSTM
input = Input(shape=(5,2))
den = LSTM(8, activation='relu')(input)
f = Flatten()(den)
den1 = Dense(16)(f)
den2 = Dense(24)(den1)
den3 = Dense(16)(den2)
den4 = Dense(8)(den3)
output = Dense(1)(den4)
model = Model(inputs=input, outputs=output)





import datetime
date = datetime.datetime.now()
print(date)                     # 2024-01-17 10:52:59.857389
date = date.strftime("%m%d_%H%M")                   # _는 str      
print(date)                     # 0117_1058
print(type(date))               # <class 'str'>

path = "..\\_data\\_save\\MCP\\"
filename = '{epoch:05d}-{val_loss:.4f}-{loss:.4f}.hdf5'            # 04d : 4자리 정수표현, 4f : 소수4번째자리까지 표현, 예) 1000_0.3333.hdf5
filepath = "".join([path, 'k26_diabetes_',date,'_', filename])
# mcp = ModelCheckpoint(monitor='val_loss', mode='min', verbose=1, save_best_only=True, filepath=filepath)    
model.compile(loss='mse', optimizer='adam')
es = EarlyStopping(monitor='val_loss', mode='min',patience=100, verbose= 1, restore_best_weights=True) 
start_time = time.time()
hist = model.fit(X_train, y_train, epochs=500, batch_size=10, validation_split=0.2, callbacks=[es])
end_time = time.time()

loss = model.evaluate(X_test, y_test)
y_predict = model.predict(X_test)
r2 = r2_score(y_test, y_predict)
def RMSE(aaa, bbb):
    np.sqrt(mean_squared_error(aaa, bbb))
    return np.sqrt(mean_squared_error(aaa, bbb))
rmse = RMSE(y_test, y_predict)
print("rmse : ", rmse)
print("걸린 시간 : ", round(end_time - start_time,2), "초")
# cpu : 4.94
# gpu : 9.9
# dnn 55.783
# cnn 49.921
# LSTM : 64.531