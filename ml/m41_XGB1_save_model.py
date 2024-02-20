from xgboost import XGBClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score
import numpy as np
import pandas as pd
param = {
            'n_estimators' : 5000,                # default 100 / 1~inf / 정수
            'learning_rate' : 0.001,           # default 0.3 / 0~1 / eta
            'max_depth' :  10 ,                 # default 6 / 0~inf / 정수
}
X, y = load_digits(return_X_y=True)
print(X.shape, y.shape) # 569, 30
# print(np.unique(y, return_counts=True)) # binary ce

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

xgb = XGBClassifier(random_state=42, )
xgb.set_params(**param, early_stopping_rounds=30)

xgb.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=1,)
print(xgb.get_params())
print("model.score : ", xgb.score(X_test, y_test))
y_pred = xgb.predict(X_test)
f1 = f1_score(y_test, y_pred, average='macro')
print("f1 : ", f1)

acc = accuracy_score(y_test, y_pred)
print("acc : ", acc)

# model.score :  0.9583333333333334
# f1 :  0.9582341019004555
# acc :  0.9583333333333334
#################################################################################
# import pickle
# import joblib

# # path = "../_data/_save/_pickle_test/"
# path = "../_data/_save/_joblib_test/"
# # pickle.dump(xgb, open(path + 'm39_pickle1_save.dat', 'wb'))
# joblib.dump(xgb, path + 'm40_joblib1_save.dat')
xgb.save_model("../_data/_save/m41_XGB1_save_model.dat")