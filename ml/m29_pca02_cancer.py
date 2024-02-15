# 스케일링 후 PCA 후 train_test_split
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import sklearn as sk
print(sk.__version__)

datasets = load_breast_cancer()
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
for i  in range(length - 1, -1, -1):
    pca = PCA(n_components=length-i)
    X_pca = pca.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42, stratify=y)

    model = RandomForestClassifier(random_state=777)
    model.fit(X_train, y_train)
    result = model.score(X_test, y_test)
    print(f"[n_components={length-i}] model.score : ", result)
    print(X_train.shape)
EVR = np.cumsum(pca.explained_variance_ratio_)
print(EVR)
'''
[그냥] model.score :  0.9473684210526315
(455, 30)
[차원 - 1] model.score :  0.9035087719298246
(455, 29)
[차원 - 2] model.score :  0.9122807017543859
(455, 28)
[차원 - 3] model.score :  0.8947368421052632
(455, 27)
'''