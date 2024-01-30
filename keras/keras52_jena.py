import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, GRU, Dense
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from keras.callbacks import EarlyStopping


########### 1. data ##########
                # path = "..//_data//kaggle//jena//jena_climate_2009_2016.csv"
                # dataset = pd.read_csv(path,index_col=0)


path = "..//_data//kaggle//jena//"
X = np.load(path + "jena_X.npy")        # timestep = 3
y = np.load(path + "jena_y.npy")

# print(X.shape)  # (420548, 3, 14)
# print(y.shape)  # (420548, )

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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2 , random_state=42, shuffle=False)

############ 2. model ###################
model = Sequential()
model.add(LSTM(30, input_shape=(3,14)))
model.add(Dense(64, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1))

model.summary()

############ 3. compile ################
model.compile(loss='mse', optimizer='adam')
es = EarlyStopping(monitor='val_loss', mode='min', patience=100, restore_best_weights=True)
model.fit(X_train, y_train, epochs=3000, batch_size=1000, validation_split=0.2, callbacks=[es])

############ 4. eva ###################
res = model.evaluate(X_test, y_test)
y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
print("loss : ", res)
print("r2 : ", r2)

# LSTM
# minmax scaleing 한거
# loss :  1.2295608030399308e-05
# r2 :  0.9993177289600603

# 이전 상황 + 풍속 이상치 -9999 처리 한거
# loss :  1.5911786249489523e-05
# r2 :  0.9991170715000118

# 이전 상황 + 컬럼 2개 log 씌운 거
# loss :  1.3533465789805632e-05
# r2 :  0.9992490419095685




# GRU
# LSTM 마지막꺼와 전처리 같음.
# loss :  1.8783706764224917e-05
# r2 :  0.9989577108626988

##################여기까진 early stopping이 아닌, 모든 epoch를 다 통과함. 여기부턴 epochs를 늘리고 earlystop을 통해 나온 값