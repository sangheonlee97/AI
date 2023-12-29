import tensorflow as tf                                         # tensorflow를 가져오고, tf라고 줄여서 쓴다.
#print(tf.__version__)                                          # 2.15.0
from keras.models import Sequential
from keras.layers import Dense
import numpy as np                                              # numpy나 pandas를 써야 속도가 빠르다.


#1. 데이터
x = np.array([1,2,3])
y = np.array([1,2,3])


#2. 모델구성
model = Sequential()                                            # 순차적인 모델을 만든다
model.add(Dense(1,input_dim=1))                                 # 밀집도를 ~~로 설정. input_dim -> 입력값이 1가지, 1 -> 출력값이 1가지


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')                     # mse 손실함수는 제곱값으로 사용, adam은 그냥 써라
model.fit(x, y, epochs=1000)                                    # 최적의 웨이트가 생성, epochs= 훈련 횟수

#4. 평가, 예측
loss = model.evaluate(x, y)
print("로스 = ", loss)
result = model.predict([4])
print("4의 예측값 = ", result)