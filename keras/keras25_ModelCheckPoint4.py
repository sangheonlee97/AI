# 세이브 파일명 만들기
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from keras.models import Sequential, save_model, load_model
from keras.layers import Dense
from sklearn.metrics import r2_score, accuracy_score
import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping, ModelCheckpoint


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
# model.load_weights("..//_data//_save//keras24_5_save_weights2.h5")
# model.save("..//_data//_save//keras24_save_model.h5")
# model.save_weights("..//_data//_save//keras24_5_save_weights1.h5")
# model = load_model("..//_data//_save//keras24_save_model.h5")
# model = load_model("..//_data//_save//keras24_3_save_model2.h5")

model.summary()



from keras.callbacks import EarlyStopping
import datetime
dated = datetime.datetime.now()
dated = dated.strftime("%m%d_%H%M")
print(dated)
path='..//_data//_save//MCP//'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'
filepath = "".join([path, dated, "_", filename])


es = EarlyStopping(monitor='val_loss',
                   patience=10,
                   verbose=1,
                   mode='min',
                   restore_best_weights=True
                   )

mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, filepath=filepath)
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=100, validation_split=0.2, callbacks=[es,mcp])
model.save("..//_data//_save//keras25.h5")

# model = load_model('..//_data//_save//MCP//keras25_MCP1.hdf5')
print("+++++++++++++++++++++++++++ 1. 기본 출력 +++++++++++++++++++++++++++++")
y_pred = model.predict(X_test)

loss = model.evaluate(X_test, y_test)

r2 = r2_score(y_test, y_pred)

print("loss : ", loss)
print("r2 : ", r2)

print("+++++++++++++++++++++++++++ 2. load_model 출력 +++++++++++++++++++++++")
model2 = load_model("..//_data//_save//keras25.h5")
y_pred = model2.predict(X_test)

loss = model2.evaluate(X_test, y_test)

r2 = r2_score(y_test, y_pred)

print("loss : ", loss)
print("r2 : ", r2)

print("+++++++++++++++++++++++++++ 3. MCP 출력 +++++++++++++++++++++++")
model3 = load_model("..//_data//_save//MCP//keras25_MCP1.hdf5")
y_pred = model3.predict(X_test)

loss = model3.evaluate(X_test, y_test)

r2 = r2_score(y_test, y_pred)

print("loss : ", loss)
print("r2 : ", r2)