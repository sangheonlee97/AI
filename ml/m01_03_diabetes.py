from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import r2_score
from sklearn.svm import LinearSVC

datasets = load_diabetes()
X = datasets.data
y = datasets.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=1226)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


from sklearn.svm import LinearSVC
model = LinearSVC()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
model.score(X_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("r2 : ", r2)

import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

