from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, StratifiedKFold
import numpy as np
import pandas as pd
datasets = load_digits()
X = datasets.data
y = datasets.target
parameters = [
    {"n_estimators": [100, 200], "max_depth": [6, 10, 12], "min_samples_leaf": [3, 10]},
    {"max_depth": [6, 8, 10, 12], "min_samples_leaf": [3, 5, 7, 10]},
    {"min_samples_leaf": [3, 5, 7, 10], "min_samples_split": [2, 3, 5, 10]},
    {"min_samples_split": [2, 3, 5, 10]},
]
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42, test_size=0.2)
kfold = StratifiedKFold(5, shuffle=True, random_state=42)
print(X.shape)
# print(np.unique(y, return_counts=True))
print(pd.value_counts(y, sort=False))

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(random_state=42, n_jobs=-1)
# gridmodel = GridSearchCV(model, param_grid=parameters, cv=kfold, refit=True, verbose=1)
# gridmodel.fit(X_train, y_train)
randommodel = RandomizedSearchCV(model, parameters, cv=kfold, refit=True, verbose=1)
randommodel.fit(X_train, y_train)
score = randommodel.score(X_test, y_test)
print(pd.DataFrame(randommodel.cv_results_))
print("score : ", score)
# score :  0.9611111111111111
# grid score :  0.9666666666666667
# randomized score :  0.9611111111111111