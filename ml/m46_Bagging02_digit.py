import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.metrics import accuracy_score, f1_score

X, y = load_digits(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42, stratify=y)

ms = MinMaxScaler()
X_train = ms.fit_transform(X_train)
X_test = ms.transform(X_test)


# model = RandomForestClassifier(random_state=42, n_estimators=10) 
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)
# print("acc : ", model.score(X_test, y_test))
# f1 = f1_score(y_test, y_pred, average='macro')
# print("f1 : ", f1)



# model = BaggingClassifier(RandomForestClassifier(random_state=42), n_estimators=10, random_state=42, verbose=1, n_jobs=-3, bootstrap=True)
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)
# print("acc : ", model.score(X_test, y_test))
# f1 = f1_score(y_test, y_pred, average='macro')
# print("f1 : ", f1)

model = BaggingClassifier(RandomForestClassifier(random_state=42), n_estimators=10, random_state=777, verbose=1, n_jobs=-3, bootstrap=True)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("acc : ", model.score(X_test, y_test))
f1 = f1_score(y_test, y_pred, average='macro')
print("f1 : ", f1)



# 그냥
# acc :  0.9361111111111111
# f1 :  0.9343648679787429

# True
# acc :  0.9694444444444444
# f1 :  0.9690485028684834


# False
# acc :  0.9694444444444444
# f1 :  0.9689455762247923