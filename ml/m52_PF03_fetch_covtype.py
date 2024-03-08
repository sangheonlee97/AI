from sklearn.datasets import fetch_covtype
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.preprocessing import PolynomialFeatures

X, y = fetch_covtype(return_X_y=True)
y = y - 1
p = PolynomialFeatures(degree=2)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y, test_size=.1)


# X_p = p.fit_transform(X)
# X_train, X_test, y_train, y_test = train_test_split(X_p, y, random_state=42, stratify=y, test_size=.1)

m = XGBClassifier(random_state= 42)

m.fit(X_train,y_train)

print('acc : ', m.score(X_test, y_test))


# acc :  0.8684038415200853
# acc :  0.8844790196550893