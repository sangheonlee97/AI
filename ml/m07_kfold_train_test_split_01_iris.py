import numpy as np
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score, cross_val_predict

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
n_splits = 5
stratifiedkfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)



model = SVC()

scores = cross_val_score(model, X_train, y_train, cv= stratifiedkfold)
print("acc : ", scores, "\n average acc : ", np.mean(scores))

pred = cross_val_predict(model, X_test, y_test, cv=stratifiedkfold)
print(pred)
print(y_test)