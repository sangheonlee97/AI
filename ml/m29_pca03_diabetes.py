# 스케일링 후 PCA 후 train_test_split
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import sklearn as sk
print(sk.__version__)

datasets = load_diabetes()
X = datasets['data']
y = datasets.target
print(X.shape, y.shape) # 150, 4


scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(random_state=777)
model.fit(X_train, y_train)
result = model.score(X_test, y_test)
print("[그냥] model.score : ", result)
print(X_train.shape)


length = X.shape[1]
for i  in range(length - 1 , -1, -1):
    pca = PCA(n_components=length-i)
    X_pca = pca.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(random_state=777)
    model.fit(X_train, y_train)
    result = model.score(X_test, y_test)
    print(f"[n_components={length - i}] model.score : ", result)
    print(X_train.shape)


evr = pca.explained_variance_ratio_
print(np.cumsum(evr))

'''
[그냥] model.score :  0.4353792704303845
(353, 10)
[n_components=1] model.score :  -0.12566474926286708
(353, 1)
[n_components=2] model.score :  0.15265343406885368
(353, 2)
[n_components=3] model.score :  0.17843346177656272
(353, 3)
[n_components=4] model.score :  0.49816509386736996
(353, 4)
[n_components=5] model.score :  0.47963422355984486
(353, 5)
[n_components=6] model.score :  0.43735640574446477
(353, 6)
[n_components=7] model.score :  0.4836347562398342
(353, 7)
[n_components=8] model.score :  0.4533319108978413
(353, 8)
[n_components=9] model.score :  0.4329284530767996
(353, 9)
[n_components=10] model.score :  0.4607168843696401
(353, 10)
'''