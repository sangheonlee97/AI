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

idx = 0
for i in range(11):
    X.iloc[:, i] = X.iloc[:, i] / max(X.iloc[:, i]) * 100
for i in range(11):
    test_csv.iloc[:, i] = test_csv.iloc[:, i] / max(test_csv.iloc[:, i]) * 100


######
le = LabelEncoder()
le.fit(X['type'])
X['type'] = le.transform(X['type'])
test_csv['type'] = le.transform(test_csv['type'])
######


y = pd.get_dummies(y) # 일케하면 0123456 으로 나옴
# y = to_categorical(y-3)
print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)
print(y_train)
a = [1,2,3,4,5,4,3,2,1]
for i, v in enumerate(a):
    print(i)
    print(v)
    print("dfafasdf")
    

'''

model = Sequential()
model.add(Dense(20, input_dim=12, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(70, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(70, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(7, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
es = EarlyStopping(monitor='val_loss', patience=200, mode='min', restore_best_weights=True, verbose=1)
model.fit(X_train, y_train, validation_split=0.3, epochs=10000, batch_size=700, callbacks=[es])

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
'''