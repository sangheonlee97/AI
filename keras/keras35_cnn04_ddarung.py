
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error,accuracy_score
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler
# 1.데이터

path = "..\\_data\\dacon\\ddarung\\"

train_csv = pd.read_csv(path + "train.csv", index_col=0)        
                                             
test_csv = pd.read_csv(path + "test.csv", index_col=0)

submission_csv = pd.read_csv(path + "submission.csv")

# train_csv = train_csv.dropna()

train_csv['hour_bef_precipitation'] = train_csv['hour_bef_precipitation'].fillna(0)
train_csv['hour_bef_pm10'] = train_csv['hour_bef_pm10'].fillna(0)
train_csv['hour_bef_pm2.5'] = train_csv['hour_bef_pm2.5'].fillna(0)
train_csv['hour_bef_windspeed'] = train_csv['hour_bef_windspeed'].fillna(0)
train_csv['hour_bef_temperature'] = train_csv['hour_bef_temperature'].fillna(train_csv['hour_bef_temperature'].mean())
train_csv['hour_bef_humidity'] = train_csv['hour_bef_humidity'].fillna(train_csv['hour_bef_humidity'].mean())
train_csv['hour_bef_visibility'] = train_csv['hour_bef_visibility'].fillna(train_csv['hour_bef_visibility'].mean())
train_csv['hour_bef_ozone'] = train_csv['hour_bef_ozone'].fillna(train_csv['hour_bef_ozone'].mean())

# train_csv['hour_bef_pm10'] = train_csv['hour_bef_pm10'].fillna(0)
# train_csv['hour_bef_pm2.5'] = train_csv['hour_bef_pm2.5'].fillna(0)
# train_csv['hour_bef_windspeed'] = train_csv['hour_bef_windspeed'].fillna(0)
# train_csv['hour_bef_temperature'] = train_csv['hour_bef_temperature'].fillna(0)
# train_csv['hour_bef_humidity'] = train_csv['hour_bef_humidity'].fillna(0)
# train_csv['hour_bef_visibility'] = train_csv['hour_bef_visibility'].fillna(0)
# train_csv['hour_bef_ozone'] = train_csv['hour_bef_ozone'].fillna(0)

# train_csv['hour_bef_temperature'] = train_csv['hour_bef_temperature'].fillna(method='backfill')
# train_csv['hour_bef_windspeed'] = train_csv['hour_bef_windspeed'].fillna(method='backfill')
# train_csv['hour_bef_humidity'] = train_csv['hour_bef_humidity'].fillna(method='backfill')
# train_csv['hour_bef_visibility'] = train_csv['hour_bef_visibility'].fillna(method='backfill')
# train_csv['hour_bef_ozone'] = train_csv['hour_bef_ozone'].fillna(method='backfill')
# train_csv['hour_bef_pm10'] = train_csv['hour_bef_pm10'].fillna(method='backfill')
# train_csv['hour_bef_pm2.5'] = train_csv['hour_bef_pm2.5'].fillna(method='backfill')

# train_csv['hour_bef_temperature'] = train_csv['hour_bef_temperature'].fillna(train_csv['hour_bef_temperature'].mean())
# train_csv['hour_bef_windspeed'] = train_csv['hour_bef_windspeed'].fillna(train_csv['hour_bef_windspeed'].mean())
# train_csv['hour_bef_humidity'] = train_csv['hour_bef_humidity'].fillna(train_csv['hour_bef_humidity'].mean())
# train_csv['hour_bef_visibility'] = train_csv['hour_bef_visibility'].fillna(train_csv['hour_bef_visibility'].mean())
# train_csv['hour_bef_ozone'] = train_csv['hour_bef_ozone'].fillna(train_csv['hour_bef_ozone'].mean())
# train_csv['hour_bef_pm10'] = train_csv['hour_bef_pm10'].fillna(train_csv['hour_bef_pm10'].mean())
# train_csv['hour_bef_pm2.5'] = train_csv['hour_bef_pm2.5'].fillna(train_csv['hour_bef_pm2.5'].mean())

test_csv = test_csv.fillna(test_csv.mean())
print(test_csv.info())      # 717 non-null

X = train_csv.drop(['count'], axis=1)

y = train_csv['count']


mas = MaxAbsScaler()
mas.fit(X)
X = mas.transform(X)
test_csv = mas.transform(test_csv)
# print(X.shape)      # 1459 9
X = X.reshape(1459, 3, 3, 1)
print(test_csv.shape)
test_csv = test_csv.reshape(715, 3, 3 , 1)
print(X.shape)      # 1459 9

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=3)      #58356
print(X_train.shape)      # 1459 9

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



# ################    RobustScaler    ##############################
# rbs = RobustScaler()
# rbs.fit(X_train)
# X_train = rbs.transform(X_train)
# X_test = rbs.transform(X_test)
# 2.모델구성

# model = Sequential()
# model.add(Dense(19,input_dim=9))
# model.add(Dense(97, activation='relu'))
# model.add(Dense(9))
# model.add(Dense(21, activation='relu'))
# model.add(Dense(19))
# model.add(Dense(8, activation='relu'))
# model.add(Dense(1))
from keras.models import Model
from keras.layers import Input
input = Input(shape=(3, 3, 1 ))
den1 = Conv2D(19, (2,2), padding='same',activation='relu')(input)
f = Flatten()(den1)
den2 = Dense(97, activation='relu')(f)
den3 = Dense(9)(den2)
den4 = Dense(21,activation='relu')(den3)
den5 = Dense(19)(den4)
den6 = Dense(8, activation='relu')(den5)
output = Dense(1)(den6)
model = Model(inputs=input, outputs=output)


# 3.컴파일, 훈련
import time
import datetime
date = datetime.datetime.now()
print(date)                     # 2024-01-17 10:52:59.857389
date = date.strftime("%m%d_%H%M")                   # _는 str      
print(date)                     # 0117_1058
print(type(date))               # <class 'str'>

path = "..\\_data\\_save\\MCP\\"
filename = '{epoch:05d}-{val_loss:.4f}-{loss:.4f}.hdf5'            # 04d : 4자리 정수표현, 4f : 소수4번째자리까지 표현, 예) 1000_0.3333.hdf5
filepath = "".join([path, 'k26_dacon_ddarung_',date,'_', filename])
# mcp = ModelCheckpoint(monitor='val_loss', mode='min', verbose=1, save_best_only=True, filepath=filepath)    
model.compile(loss='mse', optimizer='adam', metrics='accuracy')
es = EarlyStopping(monitor='val_loss', mode='min', patience=300, verbose=20, restore_best_weights=True)

start_time = time.time()

hist = model.fit(X_train, y_train, epochs=3000, batch_size=32, validation_split=0.15, verbose=2, callbacks=[es])
end_time = time.time()

# 4.평가, 예측

loss = model.evaluate(X_test, y_test)
y_submit = model.predict(test_csv)
# print(y_submit)
# print(y_submit.shape)       # (715, 1)

# print("============================================")
######### submission.csv 만들기 (count컬럼에 값만 넣어줘) #######################
submission_csv['count'] = y_submit
# print(submission_csv)
# print(submission_csv.shape)

submission_csv.to_csv(path + "submission_0116_1.csv", index=False)
y_predict = model.predict(X_test) 
def RMSE(y_test, y_predict):
    np.sqrt(mean_squared_error(y_test, y_predict))
    return np.sqrt(mean_squared_error(y_test, y_predict))
rmse = RMSE(y_test, y_predict)
print("RMSE : ", rmse)
print("로스 : ", loss)



print("걸린 시간 : ", round(end_time - start_time,2), "초")
# cpu : 38.1
# gpu : 75.96
# dnn 43.012
# cnn 41.588