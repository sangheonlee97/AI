import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score, mean_squared_log_error
from keras.callbacks import EarlyStopping,ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler

path = "..\\_data\\kaggle\\bike\\"

train_csv = pd.read_csv(path + "train.csv" , index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission_csv = pd.read_csv(path + "samplesubmission.csv")

X = train_csv.drop(['casual', 'registered', 'count'], axis=1) 
y = train_csv['count']

# print(X.shape)      #(10886, 8)
# print(y.shape)      #(10886)
mms = MinMaxScaler()
mms.fit(X)
X_train = mms.transform(X)
test_csv = mms.transform(test_csv)
X = X.values.reshape(10886,4,2)
print(test_csv.shape)
test_csv = test_csv.reshape(6493,4,2)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=713)

#############    MinMaxScaler    ##############################


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

# model = Sequential()
# model.add(Dense(19, input_shape= (8, ),activation='relu'))
# model.add(Dense(97))
# model.add(Dense(9,activation='relu'))
# model.add(Dense(21))
# model.add(Dense(16,activation='relu'))
# model.add(Dense(21))
# model.add(Dense(1, activation='relu'))
from keras.layers import Input
from keras.models import Model
from keras.layers import Conv2D, Flatten, LSTM
input = Input(shape=(4,2))
den1 = LSTM(19, activation='relu')(input)
den2 = Dense(97)(den1)
den3 = Dense(9, activation='relu')(den2)
den4 = Dense(21)(den3)
den5 = Dense(16, activation='relu')(den4)
den6 = Dense(21)(den5)
output = Dense(1, activation='relu')(den6)
model = Model(inputs=input, outputs=output)


import datetime
date = datetime.datetime.now()
# print(date)                     # 2024-01-17 10:52:59.857389
date = date.strftime("%m%d_%H%M")                   # _는 str      
# print(date)                     # 0117_1058
# print(type(date))               # <class 'str'>

path = "..\\_data\\_save\\MCP\\"
filename = '{epoch:05d}-{val_loss:.4f}-{loss:.4f}.hdf5'            # 04d : 4자리 정수표현, 4f : 소수4번째자리까지 표현, 예) 1000_0.3333.hdf5
filepath = "".join([path, 'k26_kaggle_bike_',date,'_', filename])

# mcp = ModelCheckpoint(monitor='val_loss', mode='min', verbose=1, save_best_only=True, filepath=filepath)    
model.compile(loss='mse', optimizer='adam', metrics='accuracy')
es = EarlyStopping(monitor='val_loss', mode='min', patience=300, restore_best_weights=True)

import time
start_time = time.time()


hist = model.fit(X_train, y_train, epochs= 1500, batch_size=700, validation_split=0.15,callbacks=[es])
end_time = time.time()

loss = model.evaluate(X_test, y_test)
y_submit = model.predict(test_csv)
y_predict = model.predict(X_test)
submission_csv['count'] = y_submit
print("mse : ",loss )
submission_csv.to_csv(path + "submission_0116_2.csv", index=False)

print("음수 : ", submission_csv[submission_csv['count']<0].count())

r2 = r2_score(y_test, y_predict)
def RMSLE(y_test, y_predict):
    np.sqrt(mean_squared_log_error(y_test, y_predict))
    return np.sqrt(mean_squared_log_error(y_test, y_predict))
rmsle = RMSLE(y_test, y_predict) 

print("RMSLE : ", rmsle)


# # MinMaxScaler
# RMSLE :  4.800237655708398

# # MaxAbsScaler
# RMSLE :  1.305517434934688
# # StandardScaler
# RMSLE :  1.3162207010884128

# # RobustScaler
# RMSLE :  1.2885431257772446


print("걸린 시간 : ", round(end_time - start_time,2), "초")

# cpu : 40.75
# gpu : 73.91
# dnn : 1.32
# cnn : 1.319
# LSTM : 1.326