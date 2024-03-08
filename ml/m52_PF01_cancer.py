from sklearn.datasets import load_breast_cancer
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.preprocessing import PolynomialFeatures

X, y = load_breast_cancer(return_X_y=True)
p = PolynomialFeatures(degree=2)

X_p = p.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_p, y, random_state=42, stratify=y, test_size=.1)

m = XGBClassifier(random_state= 42)

m.fit(X_train,y_train)

print('acc : ', m.score(X_test, y_test))

# acc :  0.9473684210526315
# acc :  0.9122807017543859