import numpy as np
import pandas as pd
from keras.models import Model, load_model
from keras.layers import Dense, Dropout, Input
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, RobustScaler, MinMaxScaler, StandardScaler, OrdinalEncoder
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
def save_code_to_file(filename=None):
    if filename is None:
        # 현재 스크립트의 파일명을 가져와서 확장자를 txt로 변경
        filename = os.path.splitext(os.path.basename(__file__))[0] + ".txt"
    else:
        filename = filename + ".txt"
    with open(__file__, "r") as file:
        code = file.read()
    
    with open(filename, "w") as file:
        file.write(code)
path = "..\\_data\\dacon\\daechul\\"

train_csv = pd.read_csv(path + "train.csv", index_col='ID')
test_csv = pd.read_csv(path + "test.csv", index_col='ID')
submission_csv = pd.read_csv(path + "sample_submission.csv")

############################# EDA Start ###############################
# plt.figure(figsize=(10, 10))
# loangrade = train_csv['대출등급'].value_counts()
# plt.pie(loangrade, labels=loangrade.index, autopct='%.4f', startangle=90)
# plt.show()

############################# EDA End #################################

############################# DATA Start  #############################
train_csv = train_csv.drop(labels='TRAIN_28730',axis=0) # 주택소유 상태가 any인 row 삭제

train_csv = train_csv.drop(['연체계좌수'], axis=1)  # 중요도가 낮아보이는 컬럼 삭제
test_csv = test_csv.drop(['연체계좌수'], axis=1)    # 중요도가 낮아보이는 컬럼 삭제
train_csv = train_csv.drop(['총계좌수'], axis=1)  # 중요도가 낮아보이는 컬럼 삭제
test_csv = test_csv.drop(['총계좌수'], axis=1)    # 중요도가 낮아보이는 컬럼 삭제

train_csv.loc[train_csv['근로기간']=='3','근로기간']='3 years'
test_csv.loc[test_csv['근로기간']=='3','근로기간']='3 years'
train_csv.loc[train_csv['근로기간']=='1 year','근로기간']='1 years'
test_csv.loc[test_csv['근로기간']=='1 year','근로기간']='1 years'
train_csv.loc[train_csv['근로기간']=='<1 year','근로기간']='< 1 year'
test_csv.loc[test_csv['근로기간']=='<1 year','근로기간']='< 1 year'
train_csv.loc[train_csv['근로기간']=='10+years','근로기간']='10+ years'
test_csv.loc[test_csv['근로기간']=='10+years','근로기간']='10+ years'

X = train_csv.drop(['대출등급'], axis=1)
y = train_csv['대출등급']

le_work_period = LabelEncoder()
le_work_period.fit(X['근로기간'])
X['근로기간'] = le_work_period.transform(X['근로기간'])
test_csv['근로기간'] = le_work_period.transform(test_csv['근로기간'])

# print(train_csv['주택소유상태'].value_counts()) # train 데이터에만 "any" 한개 있음,,  27번줄에서 삭제함
le_own = LabelEncoder()
le_own.fit(X['주택소유상태'])
X['주택소유상태'] = le_own.transform(X['주택소유상태'])
test_csv['주택소유상태'] = le_own.transform(test_csv['주택소유상태'])


# print(train_csv['대출목적'].value_counts())
# test_csv.iloc[34486,6] = '부채 통합'     # 결혼 -> 부채 통합 으로 임의로 바꿈 : 원래 7
test_csv.loc[test_csv['대출목적']=='결혼', '대출목적'] = '부채 통합'
le_purpose = LabelEncoder()
le_purpose.fit(X['대출목적'])
X['대출목적'] = le_purpose.transform(X['대출목적'])
test_csv['대출목적'] = le_purpose.transform(test_csv['대출목적'])

y = y.values.reshape(-1, 1)
ohe = OneHotEncoder(sparse=False)
ohe.fit(y)
y = ohe.transform(y)

le_loan_period = LabelEncoder()
le_loan_period.fit(X['대출기간'])
X['대출기간'] = le_loan_period.transform(X['대출기간'])
test_csv['대출기간'] = le_loan_period.transform(test_csv['대출기간'])
# X = X.drop(['총연체금액'],axis=1)
# X = X.drop(['연체계좌수'],axis=1)
# test_csv = test_csv.drop(['총연체금액'],axis=1)
# test_csv = test_csv.drop(['연체계좌수'],axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.07, random_state=8718650, stratify=y)

Scaler = RobustScaler(quantile_range=(26 - 1, 74 + 1))
X_train = Scaler.fit_transform(X_train)
X_test = Scaler.transform(X_test)
test_csv = Scaler.transform(test_csv)


############### 2. model ################
# model = Sequential()
# model.add(Dense(19, input_shape= (11, ),activation='relu'))
# model.add(Dense(97,activation='relu'))
# model.add(Dense(11,activation='relu'))
# model.add(Dense(10,activation='relu'))
# model.add(Dense(9,activation='relu'))
# model.add(Dense(41,activation='relu'))
# model.add(Dense(7, activation='softmax')) 

# ############### 3. compile, fit ############
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# es = EarlyStopping(monitor='val_loss', mode='min', patience=300, verbose=1, restore_best_weights=True)
# model.fit(X_train, y_train, epochs=100000, batch_size=500, validation_split=0.1, callbacks=[es])

model = load_model("..//_data//_save//dacon_loan_Rob324_0.9367.h5")

def daconsibal(vs, bs, X_train, X_test, y_train, y_test):
    es = EarlyStopping(monitor='val_loss', mode='min', patience=70, verbose=1, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=100000, batch_size=bs, validation_split=vs, callbacks=[es])

    ############### 4. evaluated, predict ##########
    results = model.evaluate(X_test, y_test)



    y_pred = model.predict(X_test)
    y_pred = ohe.inverse_transform(y_pred)
    y_test = ohe.inverse_transform(y_test)
    f1 = f1_score(y_test, y_pred, average='macro')
    print("loss : ", results[0])
    print("acc : ", results[1])
    print("f1 : ", f1)
    
    if f1 > 0.95:
        y_sub = model.predict(test_csv)
        y_sub = ohe.inverse_transform(y_sub)
        y_sub = pd.DataFrame(y_sub)


        submission_csv['대출등급'] = y_sub
        # print(sub_csv['대출등급'])
        filename = "".join(["..//_data//_save//dacon_loan_마개조1//dacon_loan_마개조_", "rs_8718650", "_bs_", str(bs), "_ts_26","_f1_", str(f1.round(4))])
        model.save(filename + ".h5")
        rfilename = "".join(["..//_data//_save//dacon_loan_마개조1//dacon_loan_마개조_", "rs_8718650", "_bs_", str(bs), "_ts_26","_f1_", str(f1.round(4)), ".csv"])
        submission_csv.to_csv(rfilename , index=False)
        save_code_to_file(filename)
    return f1

import random
while True:
    vs = random.randrange(5, 30) / 100
    bs = random.randrange(10, 5000)
    f = daconsibal(vs, bs, X_train, X_test, y_train, y_test)
    if f > 0.99:
        print("good")
        break