from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import r2_score



#1. data
datasets = load_diabetes()
X = datasets.data
y = datasets.target

print(X.shape)      # (442, 10)
print(y.shape)      # (442, )

print(datasets.feature_names)
print(datasets.DESCR)

# [실습]
# R2 0.62 이상

X_train , X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=712) #9:5.8,   

model = Sequential()
model.add(Dense(8, input_dim=10))
model.add(Dense(16))
model.add(Dense(24))
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(1))

model.compile(loss='mae', optimizer='adam')
model.fit(X_train, y_train, epochs=100, batch_size=10)

y_pred = model.predict(X_test)
model.evaluate(X_test, y_test)
r2 = r2_score(y_test, y_pred)

print("r2 : ", r2)
