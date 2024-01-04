import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split

x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,6,5,7,8,9,10])

# [검색] train과 test를 섞어서 7:3으로 나눌 수 있는 방법을 찾아라
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
print("X_train : ",X_train)
print("X_test : ",X_test)
print("y_train : ", y_train)
print("y_test : ", y_test)
