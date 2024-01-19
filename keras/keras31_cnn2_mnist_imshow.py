from tensorflow.python.keras.layers import Conv2D
import numpy as np
import pandas as pd
from keras.datasets import mnist


(X_train, y_train), (X_test, y_test) = mnist.load_data()

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
print(X_train[0])       # 5

print(pd.value_counts(y_train))
print(np.unique(y_train, return_counts=True))

import matplotlib.pyplot as plt
plt.imshow(X_train[0], 'gray')
plt.show()