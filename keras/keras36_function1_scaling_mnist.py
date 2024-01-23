from keras.layers import Conv2D, Dense, Flatten, Dropout
import numpy as np
import pandas as pd
from keras.datasets import mnist
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import OneHotEncoder


(X_train, y_train), (X_test, y_test) = mnist.load_data()
print(np.std(X_train), np.mean(X_train))
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

# not scaled         0.9882

# scaling 1_1        0.9914
X_train = X_train / 255.
X_test = X_test / 255.

# scaling 1_2        0.9886
# X_train = (X_train - 127.5) / 127.5
# X_test = (X_test - 127.5) / 127.5

# scalilng 2_1       0.9915
# X_train = X_train.reshape(60000, -1)
# X_test = X_test.reshape(10000, -1)
# scaler = MinMaxScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# scaling 2_2        0.9909
# X_train = X_train.reshape(60000, -1)
# X_test = X_test.reshape(10000, -1)
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)








y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape( -1, 1)

ohe = OneHotEncoder(sparse_output=False)
y_train = ohe.fit_transform(y_train)
y_test = ohe.transform(y_test)

X_train = X_train.reshape(60000, 28, 28, 1).astype('float32')
# X_test = X_test.reshape(10000, 28, 28, 1)
X_test = X_test.reshape(10000, 28, 28, 1).astype('float32')
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
print(np.unique(X_train, return_counts=True))
#2 data
from keras.models import Model
from keras.layers import Input
ip = Input(shape=(28,28,1))
d1 = Conv2D(20, (2, 2), activation='relu')(ip)
d2 = Conv2D(30, (4, 4), activation='relu')(d1)
d3 = Conv2D(150, (5, 5), activation='relu')(d2)
dr1 = Dropout(0.3)(d3)
d4 = Conv2D(50, (6, 6), activation='relu')(dr1)
d5 = Conv2D(60, (7, 7), activation='relu')(d4)
d6 = Conv2D(30, (8, 8), activation='relu')(d5)
f = Flatten()(d6)
d7 = Dense(10)(f)
d8 = Dense(10)(d7)
d9 = Dense(10)(d8)
op = Dense(10, activation='softmax')(d9)
model = Model(inputs=ip, outputs=op)

#3 compile, fit
from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', mode='min', patience=30, verbose=1, restore_best_weights=True )
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=1000, verbose=1, epochs=10000, validation_split=0.2 , callbacks=[es])

#3 evaluate, predict
res = model.evaluate(X_test, y_test)
print("loss : ", res[0])
print("acc : ", res[1])

