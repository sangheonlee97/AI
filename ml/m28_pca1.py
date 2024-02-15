# 스케일링 후 PCA 후 train_test_split
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import sklearn as sk
print(sk.__version__)

datasets = load_iris()
X = datasets['data']
y = datasets.target
print(X.shape, y.shape) # 150, 4

scaler = StandardScaler()
X = scaler.fit_transform(X)

pca = PCA(n_components=1)
X = pca.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state=42, stratify=y)

model = RandomForestClassifier(random_state=777)
model.fit(X_train, y_train)
result = model.score(X_test, y_test)
print("model.score : ", result)

'''
그냥 : 0.9666666666
pca(n_components=3) : 0.93333333
pca(n_components=2) : 0.9
pca(n_components=1) : 0.86666666
'''