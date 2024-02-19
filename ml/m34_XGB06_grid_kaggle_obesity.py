from xgboost import XGBClassifier, XGBRFRegressor
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, GridSearchCV, RandomizedSearchCV
# from sklearn.model_selection import HalvingGridSearchCV, HalvingRandomSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
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

path = '..//_data//kaggle//obesity//'
numpy_random_seed = 42
np.random.seed(numpy_random_seed)
def oheconcat(data, col):
    data = pd.DataFrame(data)
    temp = data.iloc[:,col]
    temp = pd.get_dummies(temp)
    data = pd.DataFrame(data)
    data = data.drop(data.columns[col], axis=1)
    data = np.concatenate([data, temp], axis=1)
    return data

train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission_csv = pd.read_csv(path + "sample_submission.csv")

roundlist = [
            'FCVC',
            'NCP',
            'CH2O',
            'FAF',
            'TUE'
]
train_csv[roundlist] = train_csv[roundlist].round()
test_csv[roundlist] = test_csv[roundlist].round()

# 1. data
X = train_csv.drop(['NObeyesdad'], axis=1)
y = train_csv['NObeyesdad']
# print(X, X.shape)   # (20758, 16)
# print(y, y.shape)   # (20758, )

# print(np.unique(y, return_counts=True)) # 7개, [2523, 3082, 2910, 3248, 4046, 2427, 2522]
#       ['Insufficient_Weight', 'Normal_Weight', 'Obesity_Type_I',
#        'Obesity_Type_II', 'Obesity_Type_III', 'Overweight_Level_I',
#        'Overweight_Level_II']

ohelist = [0, 4, 5, 9, 11, 15]
maplist = [8, 14]

mapping_dict = {'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always' :3}
for col in maplist:
    X.iloc[:, col] = X.iloc[:, col].map(mapping_dict)
    test_csv.iloc[:, col] = test_csv.iloc[:, col].map(mapping_dict)

idxcheck = 0
for col in ohelist:
    X = oheconcat(X, col - idxcheck)
    test_csv = oheconcat(test_csv, col - idxcheck)
    idxcheck += 1


le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
test_csv = scaler.transform(test_csv)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
xgb = XGBClassifier(random_state=42,)
model = RandomizedSearchCV(xgb, param, n_jobs=-3, cv=kf, random_state=42, n_iter=100)

model.fit(X_train, y_train)

print("최상의 매개변수 : ", model.best_estimator_)
print("최상의 매개변수 : ", model.best_params_)
print("최상의 score : ", model.best_score_)
print("model.score : ", model.score(X_test, y_test))

# 최상의 매개변수 :  {'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.5}
# 최상의 score :  0.904311479264368
# model.score :  0.9125722543352601