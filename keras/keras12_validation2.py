import numpy as np

#1. data
X = np.array(range(1, 17))
y = np.array(range(1, 17))

# # 잘라라!
X_train = X[:10]
X_test = X[-3:]
X_val = X[10:-3]

print(X)
print(" ")
print(X_train)
print(" ")
print(X_val)
print("  " )
print(X_test)
