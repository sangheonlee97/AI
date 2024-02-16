import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
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

train_csv = train_csv.drop(['SMOKE'], axis=1)
test_csv = test_csv.drop(['SMOKE'], axis=1)
train_csv = train_csv.drop(['TUE'], axis=1)
test_csv = test_csv.drop(['TUE'], axis=1)

roundlist = [
            'FCVC',
            'NCP',
            'CH2O',
            'FAF',
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

ohelist = [0, 4, 5, 10, 13]
lelist = [8, 12]
# ohelist = [0, 4, 5, 9, 11, 15]
# lelist = [8, 14]
# for col in lelist:
#     X, test_csv = le(X, test_csv, col)
mapping_dict = {'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always' :3}

# 'Category' 열을 매핑하여 새로운 'Mapped' 열 생성
for col in lelist:
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
submission_csv.to_csv(path + "submission_1.csv", index=False)

# score :  0.8966763005780347
# score :  0.896917148362235
# score :  0.8930635838150289


# n_splits = 5
# kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
# scores = cross_val_score(rfc, X_train, y_train, cv=kfold, n_jobs=-5)
# print("scores : ", scores)
# print("mean score : ", np.mean(scores))
# print("max score : ", np.max(scores))

# smoke 컬럼은 빼니까 성능이 더 안좋아진다..