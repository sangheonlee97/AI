import numpy as np
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score

X, y = load_iris(return_X_y=True)

n_splits = 5
StratifiedKFold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)

StratifiedKFold.split(X, y)

model = SVC()

scores = cross_val_score(model, X, y, cv= StratifiedKFold)
print("acc : ", scores, "\n average acc : ", np.mean(scores))