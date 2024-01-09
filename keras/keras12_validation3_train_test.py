import numpy as np
from sklearn.model_selection import train_test_split

#1. data
X = np.array(range(1, 17))
y = np.array(range(1, 17))

# # 잘라라!

X_train, X_test , y_train, y_test = train_test_split(X, y, test_size=0.15, shuffle=False)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size= 0.2, shuffle=False)
print(X_train)  # [ 1  2  3  4  5  6  7  8  9 10]
print(X_val)    # [11 12 13]
print(X_test)   # [14 15 16]

