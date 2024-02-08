import numpy as np
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score, cross_val_predict, GridSearchCV

best_score = [0, 999, 999]
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
n_splits = 5
StratifiedKFold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

parameters = [
    {"C":[1,10,100,1000], 'kernel':['linear'], 'degree':[3,4,5]},
    {"C":[1,10,100], 'kernel':['rbf'], 'gamma':[0.001, 0.0001]},
    {"C":[1,10,100,1000], 'kernel':['sigmoid'],'gamma':[0.001, 0.0001], 'degree':[3,4]}]

model = GridSearchCV(SVC(), parameters, cv=StratifiedKFold, verbose=1, n_jobs=10)

model.fit(X_train, y_train)

print("best 매개변수 : ", model.best_estimator_)
print("best parameters : ", model.best_params_)

print("best score : ", model.best_score_)
print("model.score : ", model.score(X_test, y_test))