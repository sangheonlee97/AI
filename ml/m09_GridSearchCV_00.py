import numpy as np
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score, cross_val_predict

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

best_score = [0, 999, 999]
for gamma in [0.001, 0.01, 0.1, 1, 10, 100]:
    for C in [0.001, 0.01, 0.1, 1, 10, 100]:
        model = SVC(gamma=gamma, C=C)
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        if best_score[0] < score:
            best_score[0] = score
            best_score[1] = gamma
            best_score[2] = C

print("best score : ", best_score[0])
print("best gamma : ", best_score[1])
print("best C     : ", best_score[2])