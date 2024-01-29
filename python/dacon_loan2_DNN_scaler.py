import numpy as np
import pandas as pd
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4, stratify=y)
# print(np.unique(X_train['연체계좌수'], return_counts=True))

Scaler = RobustScaler(quantile_range=(5,95))
X_train = Scaler.fit_transform(X_train)
X_test = Scaler.transform(X_test)
test_csv = Scaler.transform(test_csv)

# RS = [
#         '대출금액',
#         '연간소득',
#         '부채_대비_소득_비율',
#         '총계좌수',
#         '대출목적',
#         '총상환원금',
#         '총상환이자',
# ]
# Scaler = StandardScaler()
# X_train[RS] = Scaler.fit_transform(X_train[RS])
# X_test[RS] = Scaler.transform(X_test[RS])
# test_csv[RS] = Scaler.transform(test_csv[RS])
######################### DATA End ###############################




######################## MODELING Start ##########################
ip = Input(shape=(13, ))
d1 = Dense(30, activation='swish')(ip)
d2 = Dense(100, activation='swish')(d1)
d3 = Dense(30, activation='swish')(d2)
d4 = Dense(150, activation='swish')(d3)
d5 = Dense(100, activation='swish')(d4)
d6 = Dense(50, activation='swish')(d5)
op = Dense(7, activation='softmax')(d6)
model = Model(inputs=ip, outputs=op)
######################## MODELING End ############################





######################## COMPILE, FIT Start ######################
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
es = EarlyStopping(monitor='val_loss', mode='min', patience=100, verbose=1, restore_best_weights=True)
model.fit(X_train, y_train, epochs=10000, batch_size=1500, validation_split=0.2, callbacks=[es])
######################## COMPILE, FIT End ########################

######################## EVALUTATE, PREDICT Start ################
y_pred = model.predict(X_test)
y_pred = ohe.inverse_transform(y_pred)
y_test = ohe.inverse_transform(y_test)
f1 = f1_score(y_test, y_pred, average='macro')
print("f1 : ", f1)

y_sub = model.predict(test_csv)
y_sub = ohe.inverse_transform(y_sub)
y_sub = pd.DataFrame(y_sub)
######################## EVALUTATE, PREDICT End ##################




######################## SUBMISSION ##############################
submission_csv['대출등급'] = y_sub
# print(sub_csv['대출등급'])
filename = "".join(["..//_data//_save//dacon_loan_2//dacon_loan_2_", str(f1.round(4))])
model.save(filename + ".h5")
submission_csv.to_csv(path + "submisson_2.csv", index=False)
save_code_to_file(filename)