from keras.models import Sequential, Model
from keras.layers import Dense, Input
import numpy as np

#1. data
x = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 
             [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.5, 1.4, 1.3]])
y = np.array([1,2,3,4,5,6,7,8,9,10])

print(x.shape)                  # (2, 10)
print(y.shape)                  # (10, )

x = x.T                         # (10, 2)



#2. modeling (순차적)
# model = Sequential()
# model.add(Dense(10, input_dim=2))
# model.add(Dense(9))
# model.add(Dense(8))
# model.add(Dense(7))
# model.add(Dense(1))
# 열(column), 속성, feature, 차원 = 2 // 같다.
# (행 무시, 열 우선)

#2. modeling (함수형)
input = Input(shape=(2, ))
dense1 = Dense(10)(input)
dense2 = Dense(9)(dense1)
dense3 = Dense(8)(dense2)
dense4 = Dense(7)(dense3)
output = Dense(1)(dense4)
model = Model(inputs = input, outputs = output)




#3. compile, train
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=120, batch_size=1)

#4. evaluate, predict
loss = model.evaluate(x,y)

t = np.array([[10],[1.3]])
t = t.T
print("t : ", t)

result = model.predict(t)
print("loss : ",loss)
print("[[10],[1.3]] 의 예측 값 : ", result)

# [실습] : 소수점 둘째 자리까지 예측
# loss :  8.568678458686918e-05
# [[10],[1.3]] 의 예측 값 :  [[10.001596]]  