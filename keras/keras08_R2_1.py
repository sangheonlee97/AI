import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split

X = np.array([1,2,3,4,5,6,7,8,9, 10,11,12,13,14,15,16,17,18,19,20])
y = np.array([1,2,4,3,5,7,9,3,8,12,13,8,14,15,9,6,17,23,21,20])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=412)

model = Sequential()
model.add(Dense(1, input_dim=1))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, epochs=300, batch_size=1)

loss = model.evaluate(X_test, y_test)
y_pred = model.predict(X_test)

from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_pred)

print("l : ", loss)
print("r2 : ", r2)


plt.scatter(X_test, y_test)
plt.plot(X_test, y_pred, color='red')
plt.show()

# random_state=412
# l :  1.2841613292694092
# r2 :  0.9570354965110425