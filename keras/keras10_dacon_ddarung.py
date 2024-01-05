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
test_csv = test_csv.fillna(test_csv.mean())
# test_csv.fillna(value=0, inplace=True)
print(test_csv.shape)
print(test_csv.info())
################################

######## X, y를 분리 ##########
X_t = train_csv.drop(['count'], axis=1)
y_t = train_csv['count']
# X_t = train_csv.iloc[:,:-1]
# # print(X_t.shape)    # (1459, 9)

# y_t = train_csv.iloc[:,-1:]
# # print(y_t.shape)    # (1459, 1)
###############################


    

        


X_train, X_test, y_train, y_test = train_test_split(X_t, y_t, test_size=0.1, random_state=58356)
print(X_train.shape, X_test.shape)
print(y_train.shape, y_test.shape)


# 2. model

model = Sequential()
model.add(Dense(23, input_dim=9))
model.add(Dense(300))
model.add(Dense(200))
model.add(Dense(130))
model.add(Dense(200))
model.add(Dense(1))

# 3. compile
model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, epochs=500, batch_size=3)

# 4. evaluate, predict
y_submit = model.predict(test_csv)
# model.evaluate(X_test, y_test)
# r2 = r2_score(y_test, y_pred)
# print("r2 : ", r2)

####### submission.csv 만들기 ( count 컬럼에 값만 넣어주면 된다) ##########
submission_csv['count'] = y_submit
submission_csv.to_csv(path + "submission_0105.csv", index=False)