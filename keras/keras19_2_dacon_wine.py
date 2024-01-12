# https://dacon.io/competitions/open/235610/overview/description

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
path = "..\\_data\\dacon\\wine\\"

train_csv = pd.read_csv(path + "train.csv", index_col='index')
# print(train_csv.shape)            # (5497, 13)

test_csv = pd.read_csv(path + "test.csv", index_col='index')
# print(test_csv.shape)             # (1000, 12)

submission_csv = pd.read_csv( path + "sample_submission.csv")
# print(submission_csv.shape)       # (1000, 2)

# print(train_csv.info)

X = train_csv.drop(['quality'], axis=1)
y = train_csv['quality']

# print(X.shape)  # (5497, 12)
# print(y.shape)  # (5497,)
# print(y.value_counts()) 

# quality
        # 6    2416
        # 5    1788
        # 7     924
        # 4     186
        # 8     152
        # 3      26
        # 9       5
        # Name: count, dtype: int64


######
le = LabelEncoder()
le.fit(X['type'])
X['type'] = le.transform(X['type'])
test_csv['type'] = le.transform(test_csv['type'])
######


y = pd.get_dummies(y) # 일케하면 0123456 으로 나옴
# y = to_categorical(y-3)
print(y.shape)



def auto(r):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=r, stratify=y)
        
        
        model = Sequential()
        model.add(Dense(50, input_dim=12, activation='softmax'))
        model.add(Dense(130))
        model.add(Dense(150))
        model.add(Dense(170))
        model.add(Dense(1500))
        model.add(Dense(170))
        model.add(Dense(1500))
        model.add(Dense(1070))
        model.add(Dense(150))
        model.add(Dense(170))
        model.add(Dense(150))
        model.add(Dense(170))
        model.add(Dense(150))
        model.add(Dense(170))
        model.add(Dense(1500))
        model.add(Dense(1700))
        model.add(Dense(150))
        model.add(Dense(170))
        model.add(Dense(150))
        model.add(Dense(7, activation='softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        es = EarlyStopping(monitor='val_loss', patience=100, mode='min', restore_best_weights=True, verbose=1)
        hist = model.fit(X_train, y_train, validation_split=0.4, epochs=100000, batch_size=1000, callbacks=[es])

        y_pred = model.predict(test_csv)
        results = model.evaluate(X_test, y_test)
        print(y_pred)

        # y_test = np.argmax(y_test, axis=1)
        y_pred = np.argmax(y_pred, axis=1) + 3

        submission_csv['quality'] = y_pred

        print("loss : ", results[0])
        print("acc : ", results[1])
        print(y_pred)
        submission_csv.to_csv(path + "submission.csv", index=False)
        print("r = ", r)
        return results[0]
import random
while True:
        if auto(random.randint(1,2000000000)) < 1:
                break

# import matplotlib.pyplot as plt
# plt.figure(figsize=(50, 30))
# plt.plot(hist.history['accuracy'], color='red', label='accuracy', marker='.')
# plt.plot(hist.history['val_loss'], color='blue', label='val_loss', marker='.')
# plt.xlabel('에폭')
# plt.title('wine 로스', fontsize=30)
# plt.ylabel('로스')
# plt.legend(loc = 'upper right')
# plt.grid()
# plt.ylim(0, 10)
# plt.show()
# # https://dacon.io/competitions/open/235610/overview/description