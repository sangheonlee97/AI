import numpy as np
from sklearn.datasets import load_breast_cancer
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, accuracy_score
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


#1. data
datasets = load_breast_cancer()
# print(datasets.DESCR)
# print(datasets.feature_names)

X = datasets.data
y = datasets.target        # (569, 30)
print(X.shape, y.shape)     # (569, )
######################################
# print(np.unique(y)) # [0 1] ## np.unique(y, return_counts=True)
# pd_y = pd.DataFrame(y)
# print(pd_y)
# print("0 : ", pd_y[pd_y == 0].count())
# print("1 : ", pd_y[pd_y == 1].count())
######################################
a1 = np.where(y==0)
a2 = np.where(y==1)
print("0 : ", len(a1[0]) )
print("1 : ", len(a2[0]) )

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict

StratifiedKFold = StratifiedKFold(5, shuffle=True, random_state=42)
#2. model
from sklearn.utils import all_estimators
allAlgorithms = all_estimators(type_filter='classifier')
best = [0, 'no', 999]
for i, v in enumerate(allAlgorithms):
    name, algorithm = v
    print("===============================================================")

    try:
        model = algorithm()
        
        scores = cross_val_score(model, X_train, y_train, cv= StratifiedKFold)
        print("acc : ", scores, "\n average acc : ", np.mean(scores))

        y_pred = cross_val_predict(model, X_test, y_test, cv=StratifiedKFold)
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


































from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict

StratifiedKFold = StratifiedKFold(5, shuffle=True, random_state=42)
#2. model
from sklearn.utils import all_estimators
allAlgorithms = all_estimators(type_filter='classifier')
best = [0, 'no', 999]
for i, v in enumerate(allAlgorithms):
    name, algorithm = v
    print("===============================================================")

    try:
        model = algorithm()
        
        scores = cross_val_score(model, X_train, y_train, cv= StratifiedKFold)
        print("acc : ", scores, "\n average acc : ", np.mean(scores))

        y_pred = cross_val_predict(model, X_test, y_test, cv=StratifiedKFold)
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