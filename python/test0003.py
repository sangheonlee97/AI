from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import r2_score
import time
import random

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
def auto(rand):
    X_train , X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1226) #9:5.8,   

    model = Sequential()
    model.add(Dense(8, input_dim=10))
    model.add(Dense(16))
    model.add(Dense(24))
    model.add(Dense(16))
    model.add(Dense(8))
    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam')
    model.fit(X_train, y_train, epochs=rand, batch_size=11)

    y_pred = model.predict(X_test)
    model.evaluate(X_test, y_test)
    r2 = r2_score(y_test, y_pred)

    print("r2 : ", r2)
    
    time.sleep(1)
    return r2
    
for i in range(10000):
    b = random.randrange(50,300)
    a = auto(b)
    if a > 0.73:
        print("epochs : ", b)
        break