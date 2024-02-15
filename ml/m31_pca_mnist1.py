import numpy as np
import pandas as pd
from keras.datasets import mnist
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

(X_train, _), (X_test, _) = mnist.load_data()
print(X_train.shape, X_test.shape)

# X = np.append(X_train, X_test, axis=0)
X = np.concatenate([X_train, X_test], axis=0)
print(X.shape)
scaler = StandardScaler()
X = X.reshape(70000, -1)
X = scaler.fit_transform(X)
n_components = 28*28

pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X)
evr = pca.explained_variance_ratio_
evr_cumsum = np.cumsum(evr)
count = 0
for i in range(0, n_components):
    if evr_cumsum[-i-1] <= 0.95:
        break
    count += 1
    print("n_components : ", n_components - i , " , evr : ", evr_cumsum[-i-1])

print("Count : ", count)

'''
0.95  이상 : 631
0.99  이상 : 454
0.999 이상 : 299
1.0   이상 : 72
스케일링 안한거
'''

'''
0.95  : 453
0.99  : 241
0.999 : 102
1.0   : 0
스케일링 한거
'''