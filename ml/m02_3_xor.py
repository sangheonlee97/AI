import numpy as np
from sklearn.svm import LinearSVC
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
    
X_data = np.array([[0,0], [0,1], [1,0], [1,1]])
y_data = np.array([0, 1, 1, 0])
print(X_data.shape, y_data.shape)

model = LinearSVC()
# model = Perceptron()

model.fit(X_data, y_data)

acc = model.score(X_data, y_data)
y_pred = model.predict(X_data)
print(y_pred)

print("model.score : ", acc)


accc = accuracy_score(y_data, y_pred)
print("acc : ", accc)