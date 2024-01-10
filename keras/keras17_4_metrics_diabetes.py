from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import r2_score

datasets = load_diabetes()
X = datasets.data
y = datasets.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=1226)

model = Sequential()
model.add(Dense(8, input_dim=10))
model.add(Dense(16))
model.add(Dense(24))
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', mode='min', patience=10, verbose=1)

hist = model.fit(X_train, y_train, epochs= 100, batch_size=10, validation_split=0.2, callbacks=[es])

y_pred = model.predict(X_test)
model.evaluate(X_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("r2 : ", r2)

import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

plt.figure(figsize=(50, 30))
plt.plot(hist.history['loss'], c='red', label='loss', marker='.')
plt.plot(hist.history['val_loss'], c='blue', label='val_loss', marker='.')
plt.xlabel('에폭')
plt.title('당뇨 로스', fontsize=30)
plt.ylabel('로스')
plt.legend(loc = 'upper right')
plt.ylim(2500, 10000)
plt.grid()

plt.show()

