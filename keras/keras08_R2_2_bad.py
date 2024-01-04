# [실습] 고의적으로 R2 값 낮추기
# 1. R2를 음수가 아닌 0.5 이하로 만들 것 
# 2. 데이터는 건들지 말 것
# 3. 레이어는 인풋과 아웃풋 포함해서 7개 이상
# 4. batch_size=1
# 5. 히든 레이어의 노드는 10개 이상, 100개 이하
# 6. train 사이즈 75%
# 7. epoch 100번 이상



import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split

X = np.array([1,2,3,4,5,6,7,8,9, 10,11,12,13,14,15,16,17,18,19,20])
y = np.array([1,2,4,3,5,7,9,3,8,12,13,8,14,15,9,6,17,23,21,20])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=4202343)

model = Sequential()
model.add(Dense(100, input_dim=1))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, epochs=100, batch_size=1)

loss = model.evaluate(X_test, y_test)
y_pred = model.predict(X_test)

from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_pred)

print("l : ", loss)
print("r2 : ", r2)


plt.scatter(X_test, y_test)
plt.plot(X_test, y_pred, color='red')
plt.show()

# random_state=4202343
# l :  39.842308044433594
# r2 :  0.1338629647639723