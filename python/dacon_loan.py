import numpy as np
import pandas as pd
from keras.models import Sequential, save_model
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler, StandardScaler, Normalizer, RobustScaler
path = "..//_data//dacon//daechul//"
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



############## 1. data ###############

train_csv = pd.read_csv(path + "train.csv", index_col='ID')
test_csv = pd.read_csv(path + "test.csv", index_col="ID")
sub_csv = pd.read_csv(path + "sample_submission.csv")

# print(train_csv.shape)  #(96294, 14)
# print(test_csv.shape)  #(64197, 13)
# train_csv.info()


train_csv = train_csv.drop(labels='TRAIN_28730',axis=0) # 주택소유 상태가 any인 row 삭제

train_csv.loc[train_csv['근로기간']=='3','근로기간']='3 years'
test_csv.loc[test_csv['근로기간']=='3','근로기간']='3 years'
test_csv.loc[test_csv['근로기간']=='1 year','근로기간']='1 years'
train_csv.loc[train_csv['근로기간']=='1 year','근로기간']='1 years'
test_csv.loc[test_csv['근로기간']=='<1 year','근로기간']='< 1 year'
train_csv.loc[train_csv['근로기간']=='<1 year','근로기간']='< 1 year'
test_csv.loc[test_csv['근로기간']=='10+years','근로기간']='10+ years'
train_csv.loc[train_csv['근로기간']=='10+years','근로기간']='10+ years'


# train_csv = train_csv.drop(['연체계좌수'], axis=1)  # 중요도가 낮아보이는 컬럼 삭제
# test_csv = test_csv.drop(['연체계좌수'], axis=1)    # 중요도가 낮아보이는 컬럼 삭제
# train_csv = train_csv.drop(['총계좌수'], axis=1)  # 중요도가 낮아보이는 컬럼 삭제
# test_csv = test_csv.drop(['총계좌수'], axis=1)    # 중요도가 낮아보이는 컬럼 삭제
# train_csv = train_csv.drop(['주택소유상태'], axis=1)  # 중요도가 낮아보이는 컬럼 삭제
# test_csv = test_csv.drop(['주택소유상태'], axis=1)    # 중요도가 낮아보이는 컬럼 삭제




X = train_csv.drop(['대출등급'], axis=1)
y = train_csv['대출등급']



# print(train_csv['주택소유상태'].value_counts()) # train 데이터에만 "any" 한개 있음,, 
le_own = LabelEncoder()
le_own.fit(X['주택소유상태'])
X['주택소유상태'] = le_own.transform(X['주택소유상태'])
test_csv['주택소유상태'] = le_own.transform(test_csv['주택소유상태'])


# print(train_csv['대출목적'].value_counts())
test_csv.iloc[34486,7] = '부채 통합'     # 결혼 -> 부채 통합 으로 임의로 바꿈 : 원래 7
le_purpose = LabelEncoder()
le_purpose.fit(X['대출목적'])
X['대출목적'] = le_purpose.transform(X['대출목적'])
test_csv['대출목적'] = le_purpose.transform(test_csv['대출목적'])



# print(train_csv['대출등급'].value_counts())
# le_grade = LabelEncoder()
# le_grade.fit(y)
# y = le_grade.transform(train_csv['대출등급'])
# y = pd.get_dummies(y)
# y = pd.array(y)
# y = y.reshape(-1, 1)
y = y.values.reshape(-1, 1)
ohe = OneHotEncoder(sparse=False)
ohe.fit(y)
y = ohe.transform(y)







# print(train_csv['대출기간'].value_counts()) # 36, 60
# print(test_csv['대출기간'].value_counts()) # 36, 60, 0, 6
# le_loan_period = LabelEncoder()
# le_loan_period.fit(X['대출기간'])
# X['대출기간'] = le_loan_period.transform(X['대출기간'])
# test_csv['대출기간'] = le_loan_period.transform(test_csv['대출기간'])
####################
X['대출기간'] = X['대출기간'].replace({' 36 months' : 36 , ' 60 months' : 60 }).astype(int)
test_csv['대출기간'] = test_csv['대출기간'].replace({' 36 months' : 36 , ' 60 months' : 60 }).astype(int)



# print(X.shape)  #(96293, 13)
# print(y.shape)  #(96293, )

# mms = MinMaxScaler()
# mms.fit(X)
# X = mms.transform(X)
# test_csv = mms.transform(test_csv)
# loss :  0.6678425073623657
# acc :  0.798601508140564
# f1 :  0.752024962077973




# print(train_csv['근로기간'].value_counts()) # 결측치 unknown 5671 개  , 1 year이 1years 로 오기 돼있는 듯한 데이터 있음,  3도 이씀
# print(test_csv['근로기간'].value_counts()) # 결측치 unknown 3862 개
# unknown 도 라벨링 해버릴지.. unknown만 따로 빼서 학습 시킬지 고민 중.
# 우선 같이 라벨링 시도
# print(X.value_counts(['근로기간']))


# print(train_csv['근로기간'].value_counts()) # 결측치 unknown 5671 개  , 1 year이 1years 로 오기 돼있는 듯한 데이터 있음,  3도 이씀
# print(test_csv['근로기간'].value_counts()) # 결측치 unknown 3862 개

train_df = X[X['근로기간'] != 'Unknown']    # 결측치와 아닌 행 분리
test_df = X[X['근로기간'] == 'Unknown']

X2 = train_df.drop(['근로기간'], axis=1)
y2 = train_df['근로기간']
print(y2.value_counts())
y22 = y2.values.reshape(-1, 1)
ohe2 = OneHotEncoder(sparse=False)
y2_ohe = ohe2.fit_transform(y22)

X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2_ohe, test_size=0.15, stratify=y2_ohe, random_state=42)
sc2 = MinMaxScaler()
sc2.fit(X2_train)
X22_train = sc2.transform(X2_train)
X22_test = sc2.transform(X2_test)

model2 = Sequential()
model2.add(Dense(19, input_shape= (12, ),activation='relu'))
model2.add(Dense(97,))
model2.add(Dense(11,))
model2.add(Dense(10,))
model2.add(Dense(9))
model2.add(Dense(41))
model2.add(Dense(11, activation='softmax')) 

hist2 = model2.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
es2 = EarlyStopping(monitor='val_loss', mode='min', patience=100, verbose=1, restore_best_weights=True)
model2.fit(X22_train, y2_train, epochs=100000, batch_size=1000, validation_split=0.15, callbacks=[es2])

results = model2.evaluate(X22_test, y2_test)
if results[1] > 0.8:
    filename = "".join(["..//_data//_save//dacon_loan_model2_", str(results[1].round(4)),".h5"])
    model2.save(filename)
print("result : ",results[1])
y2_pred = model2.predict(X22_test)
y2_test = ohe2.inverse_transform(y2_test)
y2_pred = ohe2.inverse_transform(y2_pred)
acc2 = accuracy_score(y2_test, y2_pred)
print("acc : ", acc2)


# print(X.value_counts(['근로기간']))
le_work_period = LabelEncoder()
le_work_period.fit(X['근로기간'])
X['근로기간'] = le_work_period.transform(X['근로기간'])
test_csv['근로기간'] = le_work_period.transform(test_csv['근로기간'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=4, stratify=y)

sc = RobustScaler()
sc.fit(X_train)
X_train = sc.transform(X_train)
X_test = sc.transform(X_test)
test_csv = sc.transform(test_csv)

'''
############### 2. model ################
model = Sequential()
model.add(Dense(19, input_shape= (13, ),activation='relu'))
model.add(Dense(97,activation='relu'))
model.add(Dense(11,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(9,activation='relu'))
model.add(Dense(41,activation='relu'))
model.add(Dense(7, activation='softmax')) 

############### 3. compile, fit ############
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
es = EarlyStopping(monitor='val_loss', mode='min', patience=100, verbose=1, restore_best_weights=True)

import time
start_time = time.time()

model.fit(X_train, y_train, epochs=100000, batch_size=10000, validation_split=0.1, callbacks=[es])

end_time = time.time()

############### 4. evaluated, predict ##########
results = model.evaluate(X_test, y_test)

# gpu 61 61 80 81
# cpu 61 63

y_pred = model.predict(X_test)
y_pred = ohe.inverse_transform(y_pred)
y_test = ohe.inverse_transform(y_test)
f1 = f1_score(y_test, y_pred, average='macro')
print("loss : ", results[0])
print("acc : ", results[1])
print("f1 : ", f1)
y_sub = model.predict(test_csv)
y_sub = ohe.inverse_transform(y_sub)
y_sub = pd.DataFrame(y_sub)

print("걸린 시간 : ", round(end_time - start_time,2), "초")


if f1 > 0.9:
    filename = "".join(["..//_data//_save//dacon_loan_Rob_ikuyoit_", str(f1.round(4)),".h5"])
    filenamename = "".join(["..//_data//_save//dacon_loan_Rob_ikuyoit_", str(f1.round(4))])
    
    model.save(filename)
    save_code_to_file(filenamename)


sub_csv['대출등급'] = y_sub
# print(sub_csv['대출등급'])
sub_csv.to_csv(path + "submisson.csv", index=False)

#reset하고 첨부터 시작 0.819, 0.836, 0.828
'''