from sklearn.datasets import load_wine
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from keras.utils import to_categorical

datasets = load_wine()
X = datasets.data
y = datasets.target

print(X.shape, y.shape)     # (178, 13), (178, )
print(pd.value_counts(y))
# 1    71
# 0    59
# 2    48


######### sklearn.preprocessing의 OneHotEncoder###########
# y = y.reshape(-1, 1)
# ohe = OneHotEncoder(sparse=False)
# # ohe.fit(y)
# # y = ohe.transform(y)
# y = ohe.fit_transform(y)
############################################


######### keras.utils의 to_categorical##########
# y = to_categorical(y)
################################################


#####################################

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
from sklearn.utils import all_estimators
allAlgorithms = all_estimators(type_filter='classifier')
best = [0, 'no']
for name, algorithm in allAlgorithms:
    try:
        model = algorithm()
        
        model.fit(X_train,y_train)

        # results = model.score(X_test, y_test)

        # print("model : ", model, ", ",'score : ', results)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        print("model : ", name, ", ","acc : ", acc)
        if best[0] < acc:
            best[0] = acc
            best[1] = name
    except:
        print("qudtlsrkxdms dkfrhflwma : ", name)
print("best model : ", best[1], "\nbest acc : ", best[0])