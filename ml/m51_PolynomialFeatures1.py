import numpy as np
from sklearn.preprocessing import PolynomialFeatures

X = np.arange(12).reshape(4, 3)
print(X)

pf = PolynomialFeatures(degree = 3, include_bias=False)
X_pf = pf.fit_transform(X)
print(X_pf.shape)