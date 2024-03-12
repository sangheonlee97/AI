import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
# from xgboost import XGBClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
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
roundlist = [
            'FCVC',
            'NCP',
            'CH2O',
            'FAF',
            'TUE'
]
# train_csv[roundlist] = train_csv[roundlist].round()
# test_csv[roundlist] = test_csv[roundlist].round()

train_csv.loc[train_csv['MTRANS'] == 'Motorbike', 'MTRANS'] = 'Automobile'
test_csv.loc[test_csv['MTRANS'] == 'Motorbike', 'MTRANS'] = 'Automobile'
train_csv.loc[train_csv['MTRANS'] == 'Bike', 'MTRANS'] = 'Walking'
test_csv.loc[test_csv['MTRANS'] == 'Bike', 'MTRANS'] = 'Walking'

# 'weight'와 'height' 컬럼을 이용하여 BMI('bmi') 컬럼 생성
train_csv['Bmi'] = train_csv['Weight'] / (train_csv['Height'] / 100) ** 2
test_csv['Bmi'] = test_csv['Weight'] / (test_csv['Height'] / 100) ** 2

# for i in range(len(train_csv)):
#     train_csv.iloc[i]['FAF'] = train_csv.iloc[i]['FAF'] * (train_csv.iloc[i]['Bmi'] / 25)
# for i in range(len(test_csv)):
#     test_csv.iloc[i]['FAF'] = test_csv.iloc[i]['FAF'] * (test_csv.iloc[i]['Bmi'] / 25)

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

for col in X.columns:
    X[col] = X[col].astype('float32')
    test_csv[col] = test_csv[col].astype('float32')

X['something'] = X.iloc[:, 8] + X.iloc[:, 14]
test_csv['something'] = test_csv.iloc[:, 8] + test_csv.iloc[:, 14]
# X['something'] = (X.iloc[:, 8] + 1) * (X.iloc[:, 14] + 1)
# test_csv['something'] = (test_csv.iloc[:, 8] + 1) * (test_csv.iloc[:, 14] + 1)
le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(X_train.shape, y_train.shape)

params = {
    'n_estimators': [500, 700, 1000],
    'learning_rate': [0.008, 0.01, 0.012, 0.014, 0.016, 0.018, 0.02] ,
    'max_depth': [5, 7, 9, 11, 13, 15, 17],
    'subsample': [0.4, 0.5, 0.6, 0.7, 0.8, 1.0],
    'colsample_bytree': [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
}

# 2. model
# model = RandomizedSearchCV(XGBClassifier(), params, n_iter=30 , n_jobs=-3, cv=5, verbose=1, random_state=42)
from keras.models import Sequential
from keras.layers import Dense
print(X.shape)
print(np.unique(y, return_counts=True))
model = Sequential()
model.add(Dense(32, activation='swish', input_shape=(18,)))
model.add(Dense(64, activation='swish'))
model.add(Dense(128, activation='swish'))
model.add(Dense(256, activation='swish'))
model.add(Dense(64, activation='swish'))
model.add(Dense(7, activation='softmax'))

# 3. compile, fit
lr = 0.001

from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=10, mode='min', restore_best_weights=True)
model.compile(optimizer=Adam(learning_rate=lr), loss='sparse_categorical_crossentropy', metrics=['acc'])
model.fit(X_train, y_train, batch_size=32, epochs=1000, validation_split=0.2, callbacks=[es])

# 4. eval
sc = model.evaluate(X_test, y_test)
print("lr : {}, loss : {}, acc : {}".format(lr, sc[0], sc[1]))

# lr : 1.0, loss : 1.9472850561141968, acc : 0.19484585523605347
# lr : 0.1, loss : 1.9324415922164917, acc : 0.19484585523605347
# lr : 0.01, loss : 1.9316164255142212, acc : 0.19484585523605347
# lr : 0.001, loss : 1.2604533433914185, acc : 0.4971098303794861
# lr : 0.001, loss : 1.9307907819747925, acc : 0.19484585523605347
# lr : 0.0001, loss : 38.24764633178711, acc : 0.12138728052377701