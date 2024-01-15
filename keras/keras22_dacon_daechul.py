import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
path = "..//_data//dacon//daechul//"

############## 1. data ###############

train_csv = pd.read_csv(path + "train.csv", index_col='ID')
test_csv = pd.read_csv(path + "test.csv", index_col="ID")
sub_csv = pd.read_csv(path + "sample_submission.csv")

# print(train_csv.shape)  #(96294, 14)
# print(test_csv.shape)  #(64197, 13)
# train_csv.info()

# print(train_csv['주택소유상태'].value_counts()) # train 데이터에만 "any" 한개 있음,, 
train_csv = train_csv.drop(labels='TRAIN_28730',axis=0) # 주택소유 상태가 any인 row 삭제

le_own = LabelEncoder()
le_own.fit(train_csv['주택소유상태'])
train_csv['주택소유상태'] = le_own.transform(train_csv['주택소유상태'])
test_csv['주택소유상태'] = le_own.transform(test_csv['주택소유상태'])


# print(train_csv['대출목적'].value_counts())
test_csv.iloc[34486,7] = '이사'     # 결혼 -> 이사 로 임의로 바꿈
le_purpose = LabelEncoder()
le_purpose.fit(train_csv['대출목적'])
train_csv['대출목적'] = le_purpose.transform(train_csv['대출목적'])
test_csv['대출목적'] = le_purpose.transform(test_csv['대출목적'])



# print(train_csv['대출등급'].value_counts())
le_grade = LabelEncoder()
le_grade.fit(train_csv['대출등급'])
train_csv['대출등급'] = le_grade.transform(train_csv['대출등급'])


# print(train_csv['근로기간'].value_counts()) # 결측치 unknown 5671 개 
# print(test_csv['근로기간'].value_counts()) # 결측치 unknown 3862 개
# unknown 도 라벨링 해버릴지.. unknown만 따로 빼서 학습 시킬지 고민 중.
# 우선 같이 라벨링 시도
le_work_period = LabelEncoder()
le_work_period.fit(train_csv['근로기간'])
train_csv['근로기간'] = le_work_period.transform(train_csv['근로기간'])
test_csv['근로기간'] = le_work_period.transform(test_csv['근로기간'])


# print(train_csv['대출기간'].value_counts()) # 36, 60
# print(test_csv['대출기간'].value_counts()) # 36, 60
le_loan_period = LabelEncoder()
le_loan_period.fit(train_csv['대출기간'])
train_csv['대출기간'] = le_loan_period.transform(train_csv['대출기간'])
test_csv['대출기간'] = le_loan_period.transform(test_csv['대출기간'])   

X = train_csv.drop(['대출등급'], axis=1)
y = train_csv['대출등급']

# print(X.shape)  #(96293, 13)
# print(y.shape)  #(96293, )

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)







# ############### 2. model ################
# model = Sequential()
# model.add(Dense(50, input_shape=(13,), activation=))