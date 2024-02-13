# LabelEncoding     : 대출기간, 근로기간    ->  MinMaxScaling
# OneHotEncondig    : 주택소유, 대출목적
# 수치형 데이터     ->  StandardScaling


import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from keras.models import Model
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
ohe_own = OneHotEncoder(sparse_output=False)
t = X.loc[:,'주택소유상태']
t = t.values.reshape(-1, 1)
ohe_own.fit(t)
t = ohe_own.transform(t)
X = X.drop(['주택소유상태'], axis=1)
X = np.concatenate([X, t], axis=1)
X = pd.DataFrame(X)

t = test_csv.loc[:, '주택소유상태']
t = t.values.reshape(-1, 1)
t = ohe_own.transform(t)
test_csv = test_csv.drop(['주택소유상태'], axis=1)
test_csv = np.concatenate([test_csv, t], axis=1)
# test_csv['주택소유상태'] = ohe_own.transform(test_csv['주택소유상태'])


# print(train_csv['대출목적'].value_counts())
# test_csv.iloc[34486,6] = '부채 통합'     # 결혼 -> 부채 통합 으로 임의로 바꿈 : 원래 7
test_csv = pd.DataFrame(test_csv)
test_csv.iloc[test_csv[6]=='결혼', 6] = '부채 통합'
ohe_purpose = OneHotEncoder(sparse_output=False)
t = X.iloc[:,6]
t = t.values.reshape(-1, 1)
ohe_purpose.fit(t)
t = ohe_purpose.transform(t)
X = X.drop([6], axis=1)
X = np.concatenate([X, t], axis=1)
X = pd.DataFrame(X)

t = test_csv.iloc[:, 6]
t = t.values.reshape(-1, 1)
t = ohe_purpose.transform(t)
test_csv = test_csv.drop([6], axis=1)
test_csv = np.concatenate([test_csv, t], axis=1)
test_csv = pd.DataFrame(test_csv)

# test_csv['대출목적'] = ohe_purpose.transform(test_csv['대출목적'])

y = y.values.reshape(-1, 1)
ohe = OneHotEncoder(sparse=False, drop='first')
ohe.fit(y)
y = ohe.transform(y)

le_loan_period = LabelEncoder()
le_loan_period.fit(X[1])
X[1] = le_loan_period.transform(X[1])
test_csv[1] = le_loan_period.transform(test_csv[1])
# X = X.drop(['총연체금액'],axis=1)
# X = X.drop(['연체계좌수'],axis=1) 
# test_csv = test_csv.drop(['총연체금액'],axis=1)
# test_csv = test_csv.drop(['연체계좌수'],axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=4, stratify=y)
# print(np.unique(X_train['연체계좌수'], return_counts=True))

print(X_train.shape)
print(X_train)


# Scaler = StandardScaler()
# X_train = Scaler.fit_transform(X_train)
# X_test = Scaler.transform(X_test)
# test_csv = Scaler.transform(test_csv)

RS = [
    0,3,4,5,7,8
]
Scaler = RobustScaler()
X_train[RS] = Scaler.fit_transform(X_train[RS])
X_test[RS] = Scaler.transform(X_test[RS])
test_csv[RS] = Scaler.transform(test_csv[RS])
# X_train = X_train.astype('float32')
# X_test = X_test.astype('float32')
# test_csv = test_csv.astype('float32')

MS = [
    6,9,10
]

Scaler = MinMaxScaler()
X_train[MS] = Scaler.fit_transform(X_train[MS])
X_test[MS] = Scaler.transform(X_test[MS])
test_csv[MS] = Scaler.transform(test_csv[MS])
# X_train = X_train.astype('float32')
# X_test = X_test.astype('float32')
# test_csv = test_csv.astype('float32')


######################### DATA End ###############################




######################## MODELING Start ##########################
rfc = RandomForestClassifier(random_state=42)
rfc.fit(X_train, y_train)
######################## COMPILE, FIT End ########################

######################## EVALUTATE, PREDICT Start ################
y_pred = rfc.predict(X_test)
y_pred = ohe.inverse_transform(y_pred)
y_test = ohe.inverse_transform(y_test)
f1 = f1_score(y_test, y_pred, average='macro')
print("f1 : ", f1)

y_sub = rfc.predict(test_csv)
y_sub = ohe.inverse_transform(y_sub)
y_sub = pd.DataFrame(y_sub)
######################## EVALUTATE, PREDICT End ##################




######################## SUBMISSION ##############################
submission_csv['대출등급'] = y_sub
# print(sub_csv['대출등급'])
filename = "".join(["..//_data//dacon//daechul//dacon_loan_rf_", str(f1.round(4)), ".csv"])
# rfc.save(filename + ".h5")
submission_csv.to_csv(filename, index=False)
save_code_to_file(filename)