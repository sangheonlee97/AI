# 09_1 copy
import warnings
warnings.filterwarnings('ignore')
from sklearn.datasets import load_boston

# 현재 사이킷런 버전 1.3.0이라 보스턴 안됨. 그래서 삭제
# pip uninstall scikit-learn
# pip uninstall scikit-learn-intelex
# pip uninstall scikit-image
# pip install scikit-learn==1.1.3
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

datasets = load_boston()
print(datasets)
X = datasets.data
y = datasets.target
# print(X.shape)  # (506, 13)
# print(y.shape)  # (506, )

# print(datasets.feature_names)   # ['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO' 'B' 'LSTAT']

# print(datasets.DESCR)   # 속성

# [실습]
# train_size 0.7이상 0.9 이하
# R2 0.62 이상

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=587)

model = Sequential()
model.add(Dense(25, input_dim=13))
model.add(Dense(40))
model.add(Dense(35))
model.add(Dense(30))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

import time
start_time = time.time()

hist = model.fit(X_train, y_train, epochs=100, batch_size=4, validation_split=0.2)

def RMSE(X_train, y_train):
    return np.sqrt(mean_squared_error(X_train, y_train))

end_time = time.time()
print("걸린 시간 : ", round(end_time - start_time, 2),"초")
loss = model.evaluate(X_test, y_test)
y_pred = model.predict(X_test)
rmse = RMSE(y_test, y_pred)

from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_pred)

print("loss : ", loss)
print("r2 : ", r2)
print("rmse : ", rmse)
# random_state =  587
# epochs =  150
# batch_size =  4
# loss :  20.571069717407227
# r2 =  0.7477602886456849



print("=================hist.history========================================")
print(hist.history)
print("=================hist.history['loss']================================")
print(hist.history['loss'])
print("=================hist.history['val_loss']============================")
print(hist.history['val_loss'])
print("=====================================================================")

plt.figure(figsize=(50, 30))
plt.plot(hist.history['loss'], color='red', label='loss', marker='.')
plt.plot(hist.history['val_loss'], color='blue', label='val_loss', marker='.')
plt.xlabel('epoch')
plt.title('boston loss', fontsize=30)
plt.ylabel('loss')
plt.legend(loc = 'upper right')
plt.grid()
plt.show()