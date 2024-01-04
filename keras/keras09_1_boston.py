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



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = Sequential()
model.add(Dense(25, input_dim=13))
model.add(Dense(40))
model.add(Dense(35))
model.add(Dense(30))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

import time
start_time = time.time()

model.fit(X_train, y_train, epochs=100, batch_size=2)


end_time = time.time()
print("걸린 시간 : ", round(end_time - start_time, 2),"초")
loss = model.evaluate(X_test, y_test)
y_pred = model.predict(X_test)

from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_pred)

print("loss : ", loss)

print("r2 : ", r2)
