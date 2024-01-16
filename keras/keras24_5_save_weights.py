from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from keras.models import Sequential, load_model
from keras.layers import Dense
from sklearn.metrics import r2_score, accuracy_score
import numpy as np
import pandas as pd


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
# model.save("..//_data//_save//keras24_save_model.h5")
model.save_weights("..//_data//_save//keras24_5_save_weights1.h5")
# model = load_model("..//_data//_save//keras24_save_model.h5")
# model = load_model("..//_data//_save//keras24_3_save_model2.h5")

model.summary()
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])



from keras.callbacks import EarlyStopping

es = EarlyStopping(monitor='val_loss',
                   patience=10,
                   verbose=1,
                   mode='min'
                   )


hist = model.fit(X_train, y_train, epochs=500, batch_size=142, validation_split=0.3, callbacks=[es])


# model.save("..//_data//_save//keras24_3_save_model2.h5")
model.save_weights("..//_data//_save//keras24_5_save_weights2.h5")




y_pred = model.predict(X_test)

loss = model.evaluate(X_test, y_test)

r2 = r2_score(y_test, y_pred)

print("loss : ", loss)
print("r2 : ", r2)