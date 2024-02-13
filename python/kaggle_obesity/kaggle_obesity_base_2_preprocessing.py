import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
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
lelist = [8, 14]

mapping_dict = {'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always' :3}
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

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
test_csv = scaler.transform(test_csv)


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
submission_csv.to_csv(path + "submission_2.csv", index=False)

# 0.8976396917148363
# 0.8988439306358381
# 0.8976396917148363

# round 후
# 0.898121387283237
# 0.8959537572254336
# 0.8978805394990366
# 0.8988439306358381