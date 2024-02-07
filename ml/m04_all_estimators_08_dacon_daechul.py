import numpy as np
import pandas as pd
from keras.models import Sequential, save_model
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler, StandardScaler, Normalizer, RobustScaler
path = "..//_data//dacon//daechul//"

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
train_csv.loc[train_csv['근로기간']=='1 year','근로기간']='1 years'
test_csv.loc[test_csv['근로기간']=='1 year','근로기간']='1 years'
train_csv.loc[train_csv['근로기간']=='<1 year','근로기간']='< 1 year'
test_csv.loc[test_csv['근로기간']=='<1 year','근로기간']='< 1 year'
train_csv.loc[train_csv['근로기간']=='10+years','근로기간']='10+ years'
test_csv.loc[test_csv['근로기간']=='10+years','근로기간']='10+ years'


# train_csv = train_csv.drop(['연체계좌수'], axis=1)  # 중요도가 낮아보이는 컬럼 삭제
# test_csv = test_csv.drop(['연체계좌수'], axis=1)    # 중요도가 낮아보이는 컬럼 삭제
train_csv = train_csv.drop(['총계좌수'], axis=1)  # 중요도가 낮아보이는 컬럼 삭제
test_csv = test_csv.drop(['총계좌수'], axis=1)    # 중요도가 낮아보이는 컬럼 삭제
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
test_csv.iloc[34486,6] = '이사'     # 결혼 -> 이사 로 임의로 바꿈 : 원래 7
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
# y = y.reshape(-1, 1)\
# print(train_csv['근로기간'].value_counts()) # 결측치 unknown 5671 개  , 1 year이 1years 로 오기 돼있는 듯한 데이터 있음,  3도 이씀
# print(test_csv['근로기간'].value_counts()) # 결측치 unknown 3862 개
# unknown 도 라벨링 해버릴지.. unknown만 따로 빼서 학습 시킬지 고민 중.
# 우선 같이 라벨링 시도
# print(X.value_counts(['근로기간']))
le_work_period = LabelEncoder()
le_work_period.fit(X['근로기간'])
X['근로기간'] = le_work_period.transform(X['근로기간'])
test_csv['근로기간'] = le_work_period.transform(test_csv['근로기간'])

# X[X['근로기간'] == 1] = 0   # 1 years를 1year 로 바꿈
# X[X['근로기간'] == 5] = 6   # 3을 3 years 로 바꿈
# test_csv[test_csv['근로기간'] == 1] = 0 # # 1 years를 1year 로 바꿈
# test_csv[test_csv['근로기간'] == 5] = 6   # 3을 3 years 로 바꿈
# print(X.value_counts(['근로기간']))





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






X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=4, stratify=y)

ss = RobustScaler()
ss.fit(X_train)
X_train = ss.transform(X_train)
X_test = ss.transform(X_test)
test_csv = ss.transform(test_csv)


############### 2. model ################
from sklearn.utils import all_estimators
allAlgorithms = all_estimators(type_filter='classifier')
best = [0, 'no']
for name, algorithm in allAlgorithms:
    try:
        model = algorithm()
        
        model.fit(X_train,y_train)

        # results = model.score(X_test, y_test)

        # print("model : ", model, ", ",'score : ', results)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        print("model : ", name, ", ","acc : ", acc)
        if best[0] < acc:
            best[0] = acc
            best[1] = name
    except:
        print("qudtlsrkxdms dkfrhflwma : ", name)
print("best model : ", best[1], "\nbest acc : ", best[0])