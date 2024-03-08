from sklearn.datasets import fetch_covtype
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.preprocessing import PolynomialFeatures
path = "../_data/dacon/wine/"

#1.데이터 가져오기
train_csv = pd.read_csv(path + "train.csv", index_col=0)

train_csv['type'] = train_csv['type'].replace({"white":1, "red":0})
X = train_csv.drop(columns='quality')
y = train_csv['quality']
y = y - 3
p = PolynomialFeatures(degree=2)


# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y, test_size=.1)


X_p = p.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_p, y, random_state=42, stratify=y, test_size=.1)

m = XGBClassifier(random_state= 42)

m.fit(X_train,y_train)

print('acc : ', m.score(X_test, y_test))

# acc :  0.6418181818181818
