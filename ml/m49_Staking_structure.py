import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
X, y = load_breast_cancer(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42, stratify=y)

ms = MinMaxScaler()
X_train = ms.fit_transform(X_train)
X_test = ms.transform(X_test)
print(y_test.shape)
xgb = XGBClassifier()
rf = RandomForestClassifier()
lr = LogisticRegression()

models = [xgb, rf, lr]
train_preds = []
test_preds = []
for model in models:
    model.fit(X_train, y_train)
    train_preds.append(model.predict(X_train))
    test_preds.append(model.predict(X_test))
    print("{0} acc : {1}".format( model.__class__.__name__, model.score(X_test, y_test)))
    
    
train_preds = np.array(train_preds).T
test_preds = np.array(test_preds).T
model2 = CatBoostClassifier(verbose=0)
model2.fit(train_preds, y_train)
y_pred =  model2.predict(test_preds)
acc = accuracy_score(y_test, y_pred)
print("스태킹 결과 : ", acc)