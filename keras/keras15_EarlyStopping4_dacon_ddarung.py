# https://dacon.io/competitions/open/235576/overview/description

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import r2_score, mean_squared_error

# 1. data
path = "..\_data\dacon\ddarung\\"

train_csv = pd.read_csv(path + "train.csv", index_col='id')
# print(train_csv.shape)  # (1459, 10)

test_csv = pd.read_csv(path + "test.csv", index_col='id')
# print(test_csv.shape)   # (715, 9)

submission_csv = pd.read_csv( path + "submission.csv")
# print(submission_csv.shape) # (715, 2)

# print(train_csv.columns)
# Index(['hour', 'hour_bef_temperature', 'hour_bef_precipitation',
#        'hour_bef_windspeed', 'hour_bef_humidity', 'hour_bef_visibility',
#        'hour_bef_ozone', 'hour_bef_pm10', 'hour_bef_pm2.5', 'count'],
#       dtype='object')

# print(train_csv.info())
# print(test_csv.info())

# print(train_csv.describe())

###### 결측치 처리 1. 제거 ######
# print(train_csv.isna().sum())
train_csv = train_csv.dropna()
# print(train_csv.isna().sum())

# hour                        0 -> 0
# hour_bef_temperature        2 -> 0
# hour_bef_precipitation      2 -> 0
# hour_bef_windspeed          9 -> 0
# hour_bef_humidity           2 -> 0
# hour_bef_visibility         2 -> 0
# hour_bef_ozone             76 -> 0
# hour_bef_pm10              90 -> 0
# hour_bef_pm2.5            117 -> 0
###############################

###### 결측치 처리 2. 평균 ######
# test_csv = test_csv.fillna(test_csv.mean())
test_csv.fillna(value=0, inplace=True)
# print(test_csv.shape)
# print(test_csv.info())
#
#
################################
'''
##### 결측치 처리 3. 다른 컬럼으로 학습 #####
# 적은 수의 결측치는 그냥 평균으로 대입
#test_csv[fillna(test_csv(test_csv.mean()))
# print(test_csv.shape)                           # (715, 9)
# print(test_csv[:-3][test_csv[:-3].isna()])      # (712, 9)
# test_csv[:-3] = test_csv[:-3][test_csv[:-3].fillna(test_csv.mean())]  ###  왜 ㅇㅗ류지.
########### 결측치가 없는 부분으로 결측치에 대한 가중치 학습 \\
test_csv.iloc[:,:-3] = test_csv.iloc[:,:-3].fillna(test_csv.mean()) # 결측치가 적은 앞부분은 평균치로 대입
tX = test_csv.dropna() # 이거로 결측치가 많은 컬럼에 대해 학습시켜, 결측치를 채울거다.
tX_x = tX.iloc[:,:-3]   # 타겟 값 분리
tX_y = tX.iloc[:,-3:]

tX_x_train, tX_x_test, tX_y_train, tX_y_test = train_test_split(tX_x, tX_y, test_size=0.15, random_state=8808)
# print(tX_x_train.shape) # (572, 6)
# print(tX_y_train.shape) # (572, 3)
model_2 = Sequential()
model_2.add(Dense(10,input_dim=6))
model_2.add(Dense(10))
model_2.add(Dense(10))
model_2.add(Dense(10))
model_2.add(Dense(10))
model_2.add(Dense(3))

model_2.compile(loss='mse', optimizer='adam')
model_2.fit(tX_x_train, tX_y_train, epochs=1000, batch_size=5)

model_2.evaluate(tX_x_test, tX_y_test)
y2_pred = model_2.predict(tX_x_test)
r2_2 = r2_score(tX_y_test, y2_pred)
print("r2_2", r2_2)
########### 결측치가 없는 부분으로 결측치에 대한 가중치 학습 //

tX_test = test_csv.loc[test_csv.isna().any(axis=1)]

tX_real_x = tX_test.iloc[:,:-3]   # 타겟 값 분리
tX_real_y = tX_test.iloc[:,-3:]

y2_sub = model_2.predict(tX_real_x)
# print(tX_real_y)
tX_real_y = y2_sub
# print(tX_real_y)
# print(tX_test.iloc[:,-3:])

tX_test.iloc[:,-3:] = tX_real_y
test_csv.loc[test_csv.isna().any(axis=1)] = tX_test
######## 3개의 컬럼 결측치, 다른 컬럼으로 학습시킨 예측값으로 채움
'''
###########################################

######## X, y를 분리 ##########
X_t = train_csv.drop(['count'], axis=1)
y_t = train_csv['count']
# X_t = train_csv.iloc[:,:-1]
# # print(X_t.shape)    # (1459, 9)

# y_t = train_csv.iloc[:,-1:]
# # print(y_t.shape)    # (1459, 1)
###############################


    

        


X_train, X_test, y_train, y_test = train_test_split(X_t, y_t, test_size=0.2, random_state=242343243)
print(X_train.shape, X_test.shape)
print(y_train.shape, y_test.shape)


# 2. model

model = Sequential()
model.add(Dense(23, input_dim=9, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(1))

# 3. compile
model.compile(loss='mse', optimizer='adam')

from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', mode='min', patience=100, verbose=1, restore_best_weights=True)
hist = model.fit(X_train, y_train, epochs=10000, batch_size=50, validation_split=0.2, callbacks=[es])

# 4. evaluate, predict
y_submit = model.predict(test_csv)
y_pred = model.predict(X_test)
model.evaluate(X_test, y_test)

####### submission.csv 만들기 ( count 컬럼에 값만 넣어주면 된다) ##########
submission_csv['count'] = y_submit

# print(submission_csv[submission_csv['count'].isna()])
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
rmse = RMSE(y_test, y_pred)
print("RMSE : ", rmse)
r2 = r2_score(y_test, y_pred)
print("r2 : ", r2)



submission_csv.to_csv(path + "submission_0110_1.csv", index=False)


# import matplotlib.pyplot as plt
# plt.rcParams['font.family'] ='Malgun Gothic'
# plt.rcParams['axes.unicode_minus'] =False

# plt.figure(figsize=(50, 30))
# plt.plot(hist.history['loss'], color='red', label='loss', marker='.')
# plt.plot(hist.history['val_loss'], color='blue', label='val_loss', marker='.')
# plt.xlabel('에폭')
# plt.title('따릉이 로스', fontsize=30)
# plt.ylabel('로스')
# plt.legend(loc = 'upper right')
# plt.grid()
# # plt.ylim(0, 1000)
# plt.show()