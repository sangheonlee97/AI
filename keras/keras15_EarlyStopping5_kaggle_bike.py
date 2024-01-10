import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_log_error, mean_squared_error
path = "C:\\Study\\_data\\kaggle\\bike\\"


# 1. data
df_train = pd.read_csv(path + "train.csv", index_col=0)
df_test = pd.read_csv(path + "test.csv", index_col=0)
df_sub = pd.read_csv(path + "sampleSubmission.csv")
# print(df_train.shape)   # (10886, 11)
# print(df_test.shape)    # (6493, 8)
# print(df_sub.shape)     # (6493, 2)

# train df의 'casual', 'registered' 컬럼 삭제
df_train = df_train.drop(['casual'], axis=1)
df_train = df_train.drop(['registered'], axis=1)
# print(df_train.shape)   # (10886, 9)

# print(df_train.isna().sum()) # 0
# print(df_test.isna().sum()) # 0

# train df의 target 분리
df_train_X = df_train.drop(['count'], axis=1)
df_train_y = df_train['count']

df_train_X_train, df_train_X_test, df_train_y_train, df_train_y_test = train_test_split(df_train_X, df_train_y, test_size=0.4, shuffle=False, random_state=6544)

# 2. modeling
model = Sequential()
model.add(Dense(10, input_dim=8, activation='relu'))
model.add(Dense(150, activation='relu'))
model.add(Dense(500, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dense(500, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

# 3. compile, fit  
model.compile(loss='mse', optimizer='adam')

from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', mode='min', patience=70, verbose=1, restore_best_weights=True)

hist = model.fit(df_train_X_train, df_train_y_train, epochs=1000, batch_size=700, verbose=2, validation_split=0.4, callbacks=[es])

# 4. predict
y_pred = model.predict(df_train_X_test)
r2 = r2_score(df_train_y_test, y_pred)
print("r2 score : ",r2)

y_sub = model.predict(df_test)

df_sub['count'] = y_sub
print("음수 갯수 : ", df_sub['count'][df_sub['count']<0].count())

df_sub.to_csv(path + "submission_0109.csv", index=False)


def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
def RMSLE(y_test, y_predict):
    return np.sqrt(mean_squared_log_error(y_test, y_predict))
    
rmse = RMSE(df_train_y_test, y_pred)
print("MSE : ",model.evaluate(df_train_X_test, df_train_y_test))

print("RMSE : ", rmse)
rmsle = RMSLE(df_train_y_test, y_pred)
print("RMSLE : ", rmsle)
print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
print(hist.history)
import matplotlib.pyplot as plt
plt.rcParams['font.family'] ='Malgun Gothic'
plt.rcParams['axes.unicode_minus'] =False

plt.figure(figsize=(50, 30))
plt.plot(hist.history['loss'], color='red', label='loss', marker='.')
plt.plot(hist.history['val_loss'], color='blue', label='val_loss', marker='.')
plt.xlabel('에폭')
plt.title('캐글 자전거 로스', fontsize=30)
plt.ylabel('로스')
plt.legend(loc = 'upper right')
plt.grid()
# plt.ylim(0, 1000)
plt.show()