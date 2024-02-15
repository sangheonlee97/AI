# 스케일링 후 PCA 후 train_test_split
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import sklearn as sk
print(sk.__version__)

datasets = fetch_covtype()
X = datasets['data']
y = datasets.target
print(X.shape, y.shape) # 150, 4


scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = RandomForestClassifier(random_state=777)
model.fit(X_train, y_train)
result = model.score(X_test, y_test)
print("[그냥] model.score : ", result)
print(X_train.shape)


length = X.shape[1]
for i  in range(1, 4):
    pca = PCA(n_components=length-i)
    X = pca.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = RandomForestClassifier(random_state=777)
    model.fit(X_train, y_train)
    result = model.score(X_test, y_test)
    print(f"[차원 - {i}] model.score : ", result)
    print(X_train.shape)

'''
[그냥] model.score :  1.0
(142, 13)
[차원 - 1] model.score :  0.8888888888888888
(142, 12)
[차원 - 2] model.score :  0.9166666666666666
(142, 11)
[차원 - 3] model.score :  0.9444444444444444
(142, 10)
'''