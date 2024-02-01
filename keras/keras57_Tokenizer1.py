from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np

text = "나는 진짜 진짜 매우 매우 맛있는 밥을 엄청 마구 마구 마구 먹었다."

token = Tokenizer()
token.fit_on_texts([text])

print(token.word_index)
print(token.word_counts)

X = token.texts_to_sequences([text])
print(X)


# X1 = to_categorical(X)
# print(X1)
# print(X1.shape)
# X1 = X1[:,:,1:]
# print(X1)
# print(X1.shape)



X = np.array(X)
# print(X.shape)
X = X.reshape(-1)
print(X.shape)

# print(X.shape)
# ohe = OneHotEncoder(sparse=False)
# X2 = ohe.fit_transform(X)
# print(X2)
# print(X2.shape)


# X3 = pd.DataFrame(X)

# print("X3 : ", X3)
# print(X3.shape)
# X = pd.DataFrame(X)
X3 = pd.get_dummies(X).astype(int)

print("X3 : ", X3)
print(X3.shape)