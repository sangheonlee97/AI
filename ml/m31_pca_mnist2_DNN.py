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


# evr = pca.explained_variance_ratio_
# evr_cumsum = np.cumsum(evr)
# count = 0
# for i in range(0, n_components):
#     if evr_cumsum[-i-1] <= 0.999:
#         break
#     count += 1
#     print("n_components : ", n_components - i , " , evr : ", evr_cumsum[-i-1])

# print("Count : ", count)

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

from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
import time

ncomlist = [332, 544, 683]
results = []
for n_com in ncomlist:
    pca = PCA(n_components=n_com)
    X_pca = pca.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, stratify=y, random_state=42)
    model = Sequential()
    model.add(Dense(32, activation='swish', input_shape=(n_com, )))
    model.add(Dense(64, activation='swish'))
    model.add(Dense(128, activation='swish'))
    model.add(Dense(10, activation='softmax'))
    
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
    start_time = time.time()
    model.fit(X_train, y_train, batch_size=5000, validation_split=0.2, epochs=100)
    end_time = time.time()
    
    results.append((model.evaluate(X_test, y_test)[1], round(end_time - start_time)))
    




X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
model = Sequential()
model.add(Dense(32, activation='swish', input_shape=(784, )))
model.add(Dense(64, activation='swish'))
model.add(Dense(128, activation='swish'))
model.add(Dense(10, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
start_time = time.time()
model.fit(X_train, y_train, batch_size=5000, validation_split=0.2, epochs=100)

end_time = time.time()
re = model.evaluate(X_test, y_test)[1]
print("그냥")
print("acc : ", re)
print("걸린 시간 ", round((end_time - start_time)), "초")
for i, v in enumerate(results):
    print("결과",i+1,". PCA = ", ncomlist[i])
    print("acc : ", v[0])
    print("걸린 시간 ", v[1], "초")
    
    
'''그냥
acc :  0.9565714001655579
걸린 시간  8 초
결과 1 . PCA =  332
acc :  0.9526428580284119
걸린 시간  6 초
결과 2 . PCA =  544
acc :  0.9427857398986816
걸린 시간  7 초
결과 3 . PCA =  683
acc :  0.9407857060432434
걸린 시간  8 초'''