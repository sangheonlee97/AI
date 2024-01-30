import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
    
a = np.array(range(1,100 + 1))
X_pred = np.array(range(96, 105 + 1))
size = 5

def split_X(dataset, size):
    aaa = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : (i + size)]
        aaa.append(subset)
    return np.array(aaa)

train = split_X(a, size)
# print(bbb)
# print(bbb.shape)

X = train[:,:-1]
y = train[:, -1]
# print(X, y)
print(X.shape, y.shape)

X_pred = split_X(X_pred, size - 1)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

MMS = MinMaxScaler()
X_train = MMS.fit_transform(X_train)
X_test = MMS.transform(X_test)
X_pred = MMS.transform(X_pred)
X_train = X_train.reshape(-1, 4)
X_test = X_test.reshape(-1, 4)
X_pred = X_pred.reshape(-1,4)

model = Sequential()
# model.add(LSTM(25, input_shape=(2,2)))
model.add(Dense(64, input_shape=(4,), activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
es = EarlyStopping(monitor='val_loss', mode='min', patience=50, restore_best_weights=True)
model.fit(X_train, y_train,epochs=1000, batch_size=2, validation_split=0.1, callbacks=[es])

res = model.evaluate(X_test, y_test)

print("loss : ", res)
y_pred = model.predict(X_pred)
print("y_pred : ", y_pred.round())
print("y_pred : ", y_pred)
# loss :  0.02496100217103958
# y_pred :  [[ 99.999466]
#  [100.999725]
#  [101.99998 ]
#  [103.000244]
#  [104.00049 ]
#  [105.00075 ]
#  [106.00101 ]]