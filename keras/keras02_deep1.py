from keras.models import Sequential
from keras.layers import Dense
import random
import numpy as np


resultt = []

def deep(a,b):

#1 data
    x = np.array([1,2,3])
    y = np.array([1,2,3])




    #2 modeling
    ###[실습] 레이어의 깊이와 노드의 갯수를 이용해 4를 만들어라
    model = Sequential()
    
    model.add(Dense(2, input_dim=1))
    for i in range(1, a):
        model.add(Dense(b))
    model.add(Dense(1))



    #3 compile
    model.compile(loss="mse", optimizer="adam")
    model.fit(x,y, epochs=100)


    #4 evaluate
    loss = model.evaluate(x,y)
    print("loss : ", loss)
    result = model.predict([4])
    resultt.append(result)
    print("result : ", result)
    print("a, b", a,b)

# a = int(input("a"))
# b = int(input("b"))
for i in range(1, 100):
    deep(i,10)
    
idx = 1
for i in resultt:
    print(i, idx)
    idx += 1
    
    
# [[4.000989]] , [[3.9990513]]