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
from keras.optimizers import Adam
lr = 0.0001
model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=lr), metrics=['accuracy'])
histroy = model.fit(X_train, y_train, epochs=200, validation_split=0.2, )

# model = load_model('..//_data//_save//MCP//keras25_MCP1.hdf5')
print("+++++++++++++++++++++++++++ 1. 기본 출력 +++++++++++++++++++++++++++++")

loss = model.evaluate(X_test, y_test)

print("learning_rate : ", lr)
print("loss : ", loss[0])
print("acc : ", loss[1])

##### 100 epochs #####

# learning_rate :  1.0
# loss :  4639.08447265625
# acc :  0.9356725215911865

# learning_rate :  0.1
# loss :  0.5388486385345459
# acc :  0.9473684430122375

# learning_rate :  0.01
# loss :  0.803059458732605
# acc :  0.9415204524993896

# learning_rate :  0.001
# loss :  0.17787028849124908
# acc :  0.9707602262496948

# learning_rate :  0.0001
# loss :  0.14193931221961975
# acc :  0.9532163739204407

##### 200 epochs #####

# learning_rate :  1.0
# loss :  44229.44921875
# acc :  0.9590643048286438

# learning_rate :  0.1
# loss :  1671.5670166015625
# acc :  0.9532163739204407

# learning_rate :  0.01
# loss :  1.7911430597305298
# acc :  0.9532163739204407

# learning_rate :  0.001
# loss :  0.3120063543319702
# acc :  0.9590643048286438

# learning_rate :  0.0001
# loss :  0.11046124994754791
# acc :  0.9766082167625427