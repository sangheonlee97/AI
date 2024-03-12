# save_best_only
# restore_best_weights
# 에 대한 고찰

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from keras.models import Sequential, save_model, load_model
from keras.layers import Dense
from sklearn.metrics import r2_score, accuracy_score
import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


datasets = load_breast_cancer()
X = datasets.data
y = datasets.target


rlr = ReduceLROnPlateau(monitor='val_loss',
                        patience=10,
                        mode = 'auto',
                        verbose=1,
                        factor=0.5)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1228)

from sklearn.preprocessing import StandardScaler, MinMaxScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

print(X.shape)
print(np.unique(y, return_counts=True))

model = Sequential()
model.add(Dense(10, input_dim=30))
model.add(Dense(30))
model.add(Dense(15))
model.add(Dense(1, activation='sigmoid'))


model.summary()



from keras.callbacks import EarlyStopping


es = EarlyStopping(monitor='val_loss',
                   patience=10,
                   verbose=1,
                   mode='min',
                   restore_best_weights=True
                   )

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
histroy = model.fit(X_train, y_train, epochs=1000, validation_split=0.2, callbacks=[es, rlr])

# model = load_model('..//_data//_save//MCP//keras25_MCP1.hdf5')
print("+++++++++++++++++++++++++++ 1. 기본 출력 +++++++++++++++++++++++++++++")
y_pred = model.predict(X_test)

loss = model.evaluate(X_test, y_test)

r2 = r2_score(y_test, y_pred)

print("loss : ", loss)
print("r2 : ", r2)
