# 비지도학습에서 y는X다.
import numpy as np
from keras.datasets import mnist
import tensorflow as tf
tf.random.set_seed(888)
np.random.seed(888)


#1. 데이터
(X_train, _), (X_test, _) = mnist.load_data()

X_train = X_train.reshape(60000, 28*28).astype('float32')/255.
X_test = X_test.reshape(10000, 28*28).astype('float32')/255.

X_train_noised = X_train + np.random.normal(0, 0.1, size=X_train.shape)
X_test_noised = X_test + np.random.normal(0, 0.1, size=X_test.shape)
print(np.max(X_train_noised), np.min(X_train_noised))

X_train_noised = np.clip(X_train_noised, 0, 1)
X_test_noised = np.clip(X_test_noised, 0, 1)

print(np.max(X_train_noised), np.min(X_train_noised))


# 2.모델
from keras.models import Sequential, Model
from keras.layers import Dense, Input

def autoencoder(hidden_layer_size):
    model = Sequential()
    model.add(Dense(hidden_layer_size, activation='relu', input_shape=(28 * 28, )))
    model.add(Dense(784, activation='sigmoid'))
    return model

# hidden_size = 631
# hidden_size = 454
# hidden_size = 299
hidden_size = 72

model = autoencoder(hidden_layer_size=hidden_size)

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

model.summary()
# Total params: 101,200

# 3. 컴파일, 훈련
model.compile(optimizer='adam', loss='mse', )
# autoencoder.compile(optimizer='adam', loss='binary_crossentropy', )

model.fit(X_train_noised, X_train, epochs=30, batch_size=128, validation_split=0.2)

# 4.평가, 예측
decoded_imgs = model.predict(X_test_noised)
# evaluate는 지표를 신뢰하기 힘듬

import matplotlib.pyplot as plt

n = 10
plt.figure(figsize=(20, 4))

for i in range(n):
    ax = plt.subplot(3, n, i+1)
    plt.imshow(X_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(3, n, i + 1 + n)
    plt.imshow(X_test_noised[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(3, n, i + 1 + n + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()


