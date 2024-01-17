from sklearn.datasets import load_diabetes
from keras.models import Sequential,load_model
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import time
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler

datasets = load_diabetes()
X = datasets.data
y = datasets.target          

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=True, random_state=1226)           # 1226 713

mms = MinMaxScaler()
mms.fit(X_train)
X_train = mms.transform(X_train)
X_test = mms.transform(X_test)


# model = Sequential()
# model.add(Dense(8,input_dim=10))
# model.add(Dense(16))
# model.add(Dense(24))
# model.add(Dense(16))
# model.add(Dense(8))
# model.add(Dense(1))

# mcp = ModelCheckpoint(monitor='val_loss', mode='min', verbose=1, save_best_only=True, filepath="c:\\_data\\_save\\MCP\\keras26_MCP03_diabetes.hdf5")    
# model.compile(loss='mse', optimizer='adam')
# es = EarlyStopping(monitor='val_loss', mode='min',patience=100, verbose= 1, restore_best_weights=True) 
# start_time = time.time()
# hist = model.fit(X_train, y_train, epochs=500, batch_size=10, validation_split=0.2, callbacks=[es,mcp])
# end_time = time.time()

model = load_model("..\\_data\\_save\\MCP\\k26_3_diabetes_0117_1308_00015-3343.7598-3587.3450.hdf5")

loss = model.evaluate(X_test, y_test)
y_predict = model.predict(X_test)
r2 = r2_score(y_test, y_predict)
def RMSE(aaa, bbb):
    np.sqrt(mean_squared_error(aaa, bbb))
    return np.sqrt(mean_squared_error(aaa, bbb))
rmse = RMSE(y_test, y_predict)
print("로스 : ", loss)