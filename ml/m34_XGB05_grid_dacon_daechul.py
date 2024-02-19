from xgboost import XGBClassifier, XGBRFRegressor
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, GridSearchCV, RandomizedSearchCV
# from sklearn.model_selection import HalvingGridSearchCV, HalvingRandomSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
import numpy as np
import pandas as pd
n_splits = 5
param = {
            'n_estimators' : [100, 200, 300, 400, 500, 1000],                # default 100 / 1~inf / 정수
            'learning_rate' : [0.1, 0.2, 0.3, 0.5, 1, 0.01, 0.001],           # default 0.3 / 0~1 / eta
            'max_depth' : [None, 2, 3, 4, 5, 6, 7, 8, 9, 10] ,                 # default 6 / 0~inf / 정수
            # 'gamma' : [0,1,2,3,4,5,7,10, 100] ,                               # default 0 / 0~inf
            # 'min_child_weight' : [0, 0.01, 0.001, 0.1, 0.5, 1, 5, 10, 100],   # default 1 / 0~inf
            # 'subsample' : [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1],                    # default 1 / 0~1
            # 'colsample_bytree' : [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1],             # default 1 / 0~1
            # 'colsample_bylevel' : [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1],            # default 1 / 0~1
            # 'colsample_bynode' : [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1],             # default 1 / 0~1
            # 'reg_alpha' : [0, 0.1, 0.01, 0.001, 1, 2, 10],                    # default 0 / 0~inf / L1 절대값 가중치 규제 / alpha
            # 'reg_lambda' : [0, 0.1, 0.01, 0.001, 1, 2, 10],                   # default 1 / 0~inf / L2 제곱 가중치 규제 / lambda
            
}
# 파라미터 튜닝 맛집 : n_estimator( epochs 같은 느낌. 클수록 좋다. ), learning_rate( 작으면 작을수록 촘촘하게 학습. 작을수록 성능이 좋다. ), 

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

# y = y.values.reshape(-1, 1)
ohe = LabelEncoder()
ohe.fit(y)
y = ohe.transform(y)

le_loan_period = LabelEncoder()
le_loan_period.fit(X[1])
X[1] = le_loan_period.transform(X[1])
test_csv[1] = le_loan_period.transform(test_csv[1])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
xgb = XGBClassifier(random_state=42, )
model = RandomizedSearchCV(xgb, param, n_jobs=-3, cv=kf, random_state=42, n_iter=5)

model.fit(X_train, y_train)

print("최상의 매개변수 : ", model.best_estimator_)
print("최상의 매개변수 : ", model.best_params_)
print("최상의 score : ", model.best_score_)
print("model.score : ", model.score(X_test, y_test))

# 최상의 매개변수 :  {'n_estimators': 200, 'max_depth': 10, 'learning_rate': 0.3}
# 최상의 score :  0.8574525433791897
# model.score :  0.8619866036658186