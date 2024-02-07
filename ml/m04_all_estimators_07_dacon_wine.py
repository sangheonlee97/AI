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
from sklearn.preprocessing import StandardScaler

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





def auto(r):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=r, stratify=y)
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        
        
        from sklearn.svm import LinearSVC
        from sklearn.linear_model import Perceptron, LogisticRegression
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.ensemble import RandomForestClassifier
        models = []
        models.append(LinearSVC())
        models.append(Perceptron())
        models.append(LogisticRegression())
        models.append(KNeighborsClassifier())
        models.append(DecisionTreeClassifier())
        models.append(RandomForestClassifier())

        for model in models:
    
                model.fit(X_train, y_train)

                y_pred = model.predict(test_csv)
                results = model.score(X_test, y_test)
                print(y_pred)

                # y_test = np.argmax(y_test, axis=1)

                submission_csv['quality'] = y_pred

                print("acc : ", results)
                print(y_pred)
                submission_csv.to_csv(path + "submission.csv", index=False)
                print("r = ", r)
                return results
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