import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from imblearn.over_sampling import SMOTE
path = '..//_data//kaggle//obesity//'
def oheconcat(data, col):
    data = pd.DataFrame(data)
    temp = data.iloc[:,col]
    temp = pd.get_dummies(temp)
    data = pd.DataFrame(data)
    data = data.drop(data.columns[col], axis=1)
    data = np.concatenate([data, temp], axis=1)
    return data
numpy_random_seed = 42
np.random.seed(numpy_random_seed)

train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission_csv = pd.read_csv(path + "sample_submission.csv")

print(np.unique(train_csv[train_csv['NObeyesdad'] == 'Obesity_Type_II']['Gender'], return_counts=True))

# 1. data
X = train_csv.drop(['NObeyesdad'], axis=1)
y = train_csv['NObeyesdad']
# print(X, X.shape)   # (20758, 16)
# print(y, y.shape)   # (20758, )

# print(np.unique(y, return_counts=True)) # 7개, [2523, 3082, 2910, 3248, 4046, 2427, 2522]
#       ['Insufficient_Weight', 'Normal_Weight', 'Obesity_Type_I',
#        'Obesity_Type_II', 'Obesity_Type_III', 'Overweight_Level_I',
#        'Overweight_Level_II']

lelist = [0, 4, 5, 9, 11, 15]
maplist = [8, 14]
# for col in lelist:
#     X, test_csv = le(X, test_csv, col)
mapping_dict = {'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always' :3}

# 'Category' 열을 매핑하여 새로운 'Mapped' 열 생성
for col in maplist:
    X.iloc[:, col] = X.iloc[:, col].map(mapping_dict)
    test_csv.iloc[:, col] = test_csv.iloc[:, col].map(mapping_dict)
for col in maplist:
    print(np.unique(X.iloc[:, col], return_counts=True))

for col in lelist:
    lee = LabelEncoder()
    X.iloc[:,col] = lee.fit_transform(X.iloc[:,col])
    test_csv.iloc[:, col] = lee.transform(test_csv.iloc[:,col])

print(X)

le = LabelEncoder()
y = le.fit_transform(y)
def outliers(data_out):
    quartile_1, q2, quartile_3 = np.percentile(data_out, [15, 50, 85]) # 하위 15퍼, 상위 85퍼 이후를 이상치로 설정
    print("1사분위 : ", quartile_1)
    print("q2 : ", q2)
    print("3사분위 : ", quartile_3)
    iqr = quartile_3 - quartile_1
    print("iqr : ", iqr)
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_1 + (iqr * 1.5)
    return np.where((data_out>upper_bound) | (data_out<lower_bound))

def filldata(data, idxlist):
    for idx in idxlist:
        data.iloc[idx] = np.mean(data)
    return data

cols = X.columns
for col in cols:
    X[col] = filldata(X[col], outliers(X[col]))
# 기존
# score :  0.9019749518304432

# 처리후
# score :  0.9029383429672447

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# smt = SMOTE(random_state=42, n_jobs=-3)
# X_train, y_train = smt.fit_resample(X_train, y_train)


# 2. model
rfc = RandomForestClassifier()

# 3. compile, fit
rfc.fit(X_train, y_train)

# 4. eval
sc = rfc.score(X_test, y_test)
print("score : ", sc)
sub = rfc.predict(test_csv)
sub = le.inverse_transform(sub)

submission_csv['NObeyesdad'] = sub
submission_csv.to_csv(path + "submission.csv", index=False)

# score :  0.899325626204239
# score :  0.8995664739884393
# score :  0.8998073217726397

# n_splits = 5
# kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
# scores = cross_val_score(rfc, X_train, y_train, cv=kfold, n_jobs=-5)
# print("scores : ", scores)
# print("mean score : ", np.mean(scores))
# print("max score : ", np.max(scores))