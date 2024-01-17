
import numpy as np
import pandas as pd
from keras.models import Sequential,load_model
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error,accuracy_score
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler
# 1.데이터

path = "c:\\_data\\dacon\\ddarung\\"

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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=3)      #58356
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
mas = MaxAbsScaler()
mas.fit(X_train)
X_train = mas.transform(X_train)
X_test = mas.transform(X_test)
test_csv = mas.transform(test_csv)


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
# # 3.컴파일, 훈련
# mcp = ModelCheckpoint(monitor='val_loss', mode='min', verbose=1, save_best_only=True, filepath="c:\\_data\\_save\\MCP\\keras26_MCP04_dacon_ddarung.hdf5")    
# model.compile(loss='mse', optimizer='adam', metrics='accuracy')
# es = EarlyStopping(monitor='val_loss', mode='min', patience=300, verbose=20, restore_best_weights=True)
# hist = model.fit(X_train, y_train, epochs=3000, batch_size=32, validation_split=0.15, verbose=2, callbacks=[es,mcp])

model = load_model("..\\_data\\_save\\MCP\\k26_4_dacon_ddarung_0117_1310_00666-1223.1259-1042.2554.hdf5")


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



