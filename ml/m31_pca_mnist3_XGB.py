import numpy as np
import pandas as pd
from keras.datasets import mnist
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

(X_train, y_train), (X_test, y_test) = mnist.load_data()
print(X_train.shape, X_test.shape)

# X = np.append(X_train, X_test, axis=0)
X = np.concatenate([X_train, X_test], axis=0)
print(X.shape)
scaler = StandardScaler()
X = X.reshape(70000, -1)
X = scaler.fit_transform(X)

y = np.concatenate([y_train, y_test], axis=0)
y = y.reshape(70000, -1)


from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
import time
param = [
    {"n_estimators":[100,200,300], "learning_rate" : [0.1, 0.3, 0.001, 0.01], "max_depth":[4,5,6] },
    {"n_estimators":[90,100,110], "learning_rate":[0.1, 0.001, 0.01], "max_depth":[4,5,6], "colsample_bytree":[0.6, 0.9, 1]},
    {"n_estimators":[90,110], "learning_rate":[0.1, 0.001, 0.5], "max_depth":[4,5,6], "colsample_bytree":[0.6, 0.9, 1], "colsample_bylevel":[0.6, 0.7, 0.9]}
]
kf = StratifiedKFold(3, shuffle=True, random_state=42)
xgb = XGBClassifier(tree_method='hist', device='cuda')
ncomlist = [332, 544, 683]
results = []
for n_com in ncomlist:
    pca = PCA(n_components=n_com)
    X_pca = pca.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, stratify=y, random_state=42)
    model = RandomizedSearchCV(xgb, param, n_jobs=-2, cv=kf, random_state=42,verbose=1)
    
    start_time = time.time()
    model.fit(X_train, y_train)
    end_time = time.time()
    
    results.append((model.score(X_test, y_test), round(end_time - start_time)))
    




X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
model = RandomizedSearchCV(xgb, param, n_jobs=-2, cv=kf, random_state=42,verbose=1)

start_time = time.time()
model.fit(X_train, y_train)
end_time = time.time()

re = model.score(X_test, y_test)
print("그냥")
print("acc : ", re)
print("걸린 시간 ", round((end_time - start_time)), "초")
for i, v in enumerate(results):
    print("결과",i+1,". PCA = ", ncomlist[i])
    print("acc : ", v[0])
    print("걸린 시간 ", v[1], "초")