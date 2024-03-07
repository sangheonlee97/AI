
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
X, y = load_digits(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=777, stratify=y)

ms = MinMaxScaler()
X_train = ms.fit_transform(X_train)
X_test = ms.transform(X_test)
print(y_test.shape)
xgb = XGBClassifier()
rf = RandomForestClassifier()
lr = LogisticRegression()

model = StackingClassifier(estimators=[('lr', lr),('rf', rf),('xgb', xgb)],
                           final_estimator=CatBoostClassifier(verbose=0),
                           )

model.fit(X_train, y_train)
print("acc : ", model.score(X_test,y_test))