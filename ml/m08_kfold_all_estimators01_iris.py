from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.utils import all_estimators
import numpy as np
import warnings
warnings.filterwarnings('ignore')



datasets = load_iris()
X = datasets.data
y = datasets.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y )
n_splits = 5
stratifiedkfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)
from sklearn.utils import all_estimators
allAlgorithms = all_estimators(type_filter='classifier')
best = [0, 'no', 999]
for i, v in enumerate(allAlgorithms):
    name, algorithm = v
    print("===============================================================")

    try:
        model = algorithm()
        
        scores = cross_val_score(model, X_train, y_train, cv= stratifiedkfold)
        print("acc : ", scores, "\n average acc : ", np.mean(scores))

        y_pred = cross_val_predict(model, X_test, y_test, cv=stratifiedkfold)
        print(y_pred)
        print(y_test)

        acc = accuracy_score(y_test, y_pred)
        print("model : ", name, ", ","acc : ", acc)
        if best[0] < acc:
            best[0] = acc
            best[1] = name
            best[2] = i
    except:
        print("바보 : ", name)
    print("===============================================================")
    
print("best model : ", best[1], ", idx : ", best[2], "\nbest acc : ", best[0])