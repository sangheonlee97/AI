from sklearn.datasets import fetch_california_housing
from keras.models import Sequential, load_model
from keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split
import time
import numpy as np
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler
from keras.callbacks import EarlyStopping, ModelCheckpoint

datasets = fetch_california_housing()
X = datasets.data
y = datasets.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, shuffle=True, random_state=33)

mas = MaxAbsScaler()
mas.fit(X_train)
X_train = mas.transform(X_train)
X_test = mas.transform(X_test)



# model = Sequential()
# model.add(Dense(21,input_dim=8))
# model.add(Dense(7))
# model.add(Dense(18))
# model.add(Dense(12))
# model.add(Dense(1))

                        
# mcp = ModelCheckpoint(monitor='val_loss', mode='min', verbose=1, save_best_only=True, filepath="c:\\_data\\_save\\MCP\\keras26_MCP02_california.hdf5")    
# model.compile(loss='mse', optimizer='adam')                                                             # early stopping 개념, min,max, auto
# es = EarlyStopping(monitor='val_loss', mode='min',patience=100, verbose= 1, restore_best_weights=True) 
# hist = model.fit(X_train, y_train, epochs=200, batch_size=400, validation_split=0.2, callbacks=[es,mcp])
# end_time = time.time()

model = load_model("..\\_data\\_save\\MCP\\k26_2_california_0117_1308_00196-0.5282-0.5595.hdf5")


loss = model.evaluate(X_test, y_test)
print("로스 : ", loss)

y_predict = model.predict(X_test)
result = model.predict(X)

from sklearn.metrics import r2_score, mean_squared_error
r2 = r2_score(y_test, y_predict)
def RMSE(aaa, bbb):
    np.sqrt(mean_squared_error(aaa, bbb))
    return np.sqrt(mean_squared_error(aaa, bbb))
rmse = RMSE(y_test, y_predict)

print("R2스코어 : ", r2)
# print("걸린시간 : ", round(end_time - start_time, 3), "초")
print("RMSE : ", rmse)
