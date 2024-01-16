from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import r2_score, accuracy_score
import numpy as np
import pandas as pd
import time


datasets = fetch_california_housing()
X = datasets.data
y = datasets.target



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1228)

from sklearn.preprocessing import StandardScaler, MinMaxScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)



model = Sequential()
model.add(Dense(10, input_dim=8))
model.add(Dense(30))
model.add(Dense(15))
model.add(Dense(1))
model.save("..//_data//_save//keras24_save_model.h5")

model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])



start_time = time.time()

from keras.callbacks import EarlyStopping

es = EarlyStopping(monitor='val_loss',
                   patience=10,
                   verbose=1,
                   mode='min'
                   )


hist = model.fit(X_train, y_train, epochs=500, batch_size=142, validation_split=0.3, callbacks=[es])

end_time = time.time()

y_pred = model.predict(X_test)

loss = model.evaluate(X_test, y_test)

r2 = r2_score(y_test, y_pred)

print("loss : ", loss)
print("r2 : ", r2)
print("걸린 시간 : ", round(end_time - start_time,2), "초")

# loss :  0.5542855858802795
# r2 :  0.5767131444516536
# random_state = 1228
# epochs = 1000
# batch_size = 142

import matplotlib.pyplot as plt
plt.rcParams['font.family'] ='Malgun Gothic'
plt.rcParams['axes.unicode_minus'] =False

# plt.figure(figsize=(50, 30))
# plt.plot(hist.history['loss'], color='red', label='loss', marker='.')
# plt.plot(hist.history['val_loss'], color='blue', label='val_loss', marker='.')
# plt.xlabel('에폭')
# plt.title('캘리포니아 로스', fontsize=30)
# plt.ylabel('로스')
# plt.legend(loc = 'upper right')
# plt.grid()
# plt.ylim(0, 1000)
# plt.show()