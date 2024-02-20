from xgboost import XGBClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score
import numpy as np
import pandas as pd
import pickle
import joblib

# path = "../_data/_save/_pickle_test/"
# path = "../_data/_save/_joblib_test/"
path = "../_data/_save/_joblib_test/"

X, y = load_digits(return_X_y=True)
print(X.shape, y.shape) # 569, 30
# print(np.unique(y, return_counts=True)) # binary ce

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# model = pickle.load(open(path + 'm39_pickle1_save.dat', 'rb'))
model = XGBClassifier()
model.load_model("../_data/_save/m41_XGB1_save_model.dat")
print("model.score : ", model.score(X_test, y_test))
y_pred = model.predict(X_test)
f1 = f1_score(y_test, y_pred, average='macro')
print("f1 : ", f1)

acc = accuracy_score(y_test, y_pred)
print("acc : ", acc)

# model.score :  0.9583333333333334
# f1 :  0.9582341019004555
# acc :  0.9583333333333334
#################################################################################
# model.score :  0.9583333333333334
# f1 :  0.9582341019004555
# acc :  0.9583333333333334