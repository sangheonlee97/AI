import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, GRU, Dense
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from keras.callbacks import EarlyStopping
import time
########### 1. data ##########
                # path = "..//_data//kaggle//jena//jena_climate_2009_2016.csv"
                # dataset = pd.read_csv(path,index_col=0)

print("load start")
path = "..//_data//kaggle//jena//"
X = np.load(path + "jena_X.npy")        # timestep = 6 * 24 * 10
y = np.load(path + "jena_y.npy")
print("load end")
print(X.shape)  # (420548, 3, 14)
print(y.shape)  # (420548, )

# (N, 2, 360, 14)


# col = dataset.columns
# for i in col:
#     print("col = ",i)
#     print("최대값 : ", np.max(dataset[i]))
#     print("최소값 : ", np.min(dataset[i]))
#     print("\n\n\n\n\n")

######################################   풍향 군집화
# X = dataset['wd (deg)']
# X.loc[ (X >= 100.80) & (X <= 324.00)] = 500.
# X.loc[ X != 500.] = 0.
# X.loc[ X == 500.] = 1.
######################################



#########시각화########
# plt.hist(X, bins=20, edgecolor='black')
# plt.title('Histogram of sqrt(VPdef (mbar))')
# plt.xlabel('sqrt(VPdef (mbar))')
# plt.ylabel('Frequency')
# plt.show()
print("split start")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2 , random_state=42, shuffle=False)
print("split end")

############ 2. model ###################
model = Sequential()
model.add(LSTM(30, input_shape=( 24 * 10, 14)))
model.add(Dense(64, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

model.summary()

############ 3. compile ################
sttime = time.time()
model.compile(loss='mse', optimizer='adam')
es = EarlyStopping(monitor='val_loss', mode='min', patience=100, restore_best_weights=True)
model.fit(X_train, y_train, epochs=3000, batch_size=3000, validation_split=0.2, callbacks=[es])
edtime = time.time()
############ 4. eva ###################
res = model.evaluate(X_test, y_test)

y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
print("loss : ", res)
print("r2 : ", r2)
print("걸린시간 : ", int(edtime - sttime))
# LSTM  전처리 싹다
# loss :  0.002523343311622739
# r2 :  0.8602245009517633
# 걸린시간 :  203 초

# LSTM NO 전처리
# loss :  10.005688667297363
# r2 :  0.8475204954112067
# 걸린시간 :  497

# LSTM 전처리 MinMax 스케일링만
# loss :  0.0026046873535960913
# r2 :  0.855718636585975
# 걸린시간 :  275

# LSTM 전처리 MinMax 스케일링, sqrt
# loss :  0.0026112545747309923
# r2 :  0.8553548446975112
# 걸린시간 :  222

# GRU
# loss :  0.002643501153215766
# r2 :  0.8535686623729211
# 걸린시간 :  207 초
################################################## 위 1시간 단위

################################################## 아래 30분 단위
# LSTM 전처리 싹다 + 40분 단위
# loss :  0.002702952129766345
# r2 :  0.8502755834110529
# 걸린시간 :  282