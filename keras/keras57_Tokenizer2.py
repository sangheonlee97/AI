from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np

text1 = "나는 진짜 진짜 매우 매우 맛있는 밥을 엄청 마구 마구 마구 먹었다."
text2 = "상헌이가 선생을 괴롭힌다. 상헌이는 못생겼다. 상헌이는 마구 마구 못생겼다."

token = Tokenizer()
token.fit_on_texts([text1, text2])

print(token.word_index)
# print(token.word_counts)
idx = np.array(list(token.word_index.values()))
X = token.texts_to_sequences([text1,text2])
X1 = np.array(X.pop(0))
X2 = np.array(X.pop(0))
print(X)
print("idx " ,idx)
idx = idx.reshape(-1, 1)

ohe = OneHotEncoder(sparse=False)
ohe.fit(idx)
X1 = X1.reshape(-1, 1)
X2 = X2.reshape(-1, 1)
X1 = ohe.transform(X1)
X2 = ohe.transform(X2)

print(X1)
print(X1.shape)
print(X2)
print(X2.shape)
# # X1 = X1[:,:,1:]
# print(X1)
# print(X1.shape)
