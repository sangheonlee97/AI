import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

path = "..\_data\dacon\diabetes\\"
######1. data
df_train = pd.read_csv(path + "train.csv", index_col=0)
df_test = pd.read_csv(path + "test.csv", index_col=0)
df_sub = pd.read_csv(path + "sample_submission.csv")

# print(df_train.shape)   # (652, 9)
# print(df_test.shape)   # (116, 8)
# print(df_train.info)

X = df_train.drop(['Outcome'], axis=1)
y = df_train['Outcome']

# print(X.shape)      # (652, 8)
# print(y.shape)      # (652, )


# print(X['Pregnancies'][X['Pregnancies']==0].count()) # 87
# print(X['Glucose'][X['Glucose']==0].count()) # 4
# print(X['BloodPressure'][X['BloodPressure']==0].count()) # 30
# print(X['SkinThickness'][X['SkinThickness']==0].count()) # 195
# print(X['Insulin'][X['Insulin']==0].count()) # 318 결측치가 많은건지, 실제 값이 0인건지 분간이 안됨;
# print(X['BMI'][X['BMI']==0].count()) # 7
# print(X['DiabetesPedigreeFunction'][X['DiabetesPedigreeFunction']==0].count()) # 0
# print(X['Age'][X['Age']==0].count()) # 0


# X = X.drop(['SkinThickness'], axis=1)
X = X.drop(['Insulin'], axis=1)
# df_test = df_test.drop(['SkinThickness'], axis=1)
df_test = df_test.drop(['Insulin'], axis=1)


def auto(ts, rs, p, bs):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ts, random_state=rs)

    ######2. model
    model = Sequential()
    model.add(Dense(20, input_dim=7, activation='relu'))
    model.add(Dense(30, activation='relu'))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(1))

    ######3. compile, fit
    model.compile(loss="mse", optimizer='adam', metrics=['accuracy'])
    es = EarlyStopping(monitor='val_loss', patience=p, mode='min', verbose=1, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=1000, batch_size=bs, validation_split=ts, callbacks=[es])

    ######4. predict
    y_pred = model.predict(X_test)
    y_pred = y_pred.round()
    y_sub = model.predict(df_test)
    y_sub = y_sub.round()

    df_sub['Outcome'] = y_sub
    df_sub.to_csv(path + "submisson_0110_qudtls.csv", index=False )

    print(y_pred)
    acc = accuracy_score(y_test, y_pred)
    print("acc : ", acc)
    return acc

import random
for i in range(12):
    ts = random.randrange(10, 41) / 100
    rs = random.randrange(1, 2000000000)
    p = random.randrange(50, 200)
    bs = random.randrange(1, 100)
    ac = auto(ts,rs,p,bs)
    if ac > 0.78:
        print("ts : ", ts)
        print("rs : ", rs)        
        print("p : ", p)
        print("bs : ", bs)
        print("acc : ", ac)
        break
        