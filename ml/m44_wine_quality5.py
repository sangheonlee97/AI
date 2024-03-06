import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
param = {
            'n_estimators' : 1000,                # default 100 / 1~inf / 정수
            'learning_rate' : 0.01,           # default 0.3 / 0~1 / eta
            'max_depth' :  3 ,                 # default 6 / 0~inf / 정수
            'gamma' : 0 ,                               # default 0 / 0~inf
            'min_child_weight' : 0,             # default 1 / 0~inf
            'subsample' : 0.4,                  # default 1 / 0~1
            'colsample_bytree' : 0.8,             # default 1 / 0~1
            'colsample_bylevel' : 0.7,            # default 1 / 0~1
            'colsample_bynode' : 1 ,             # default 1 / 0~1
            'reg_alpha' : 0,                    # default 0 / 0~inf / L1 절대값 가중치 규제 / alpha
            'reg_lambda' : 1 ,                  # default 1 / 0~inf / L2 제곱 가중치 규제 / lambda        
            'random_state' : 3377,
}
path = "../_data/dacon/wine/"

#1.데이터 가져오기
train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission_csv = pd.read_csv(path + "sample_submission.csv")

train_csv['type'] = train_csv['type'].replace({"white":1, "red":0})
test_csv['type'] = test_csv['type'].replace({"white":1, "red":0})
X = train_csv.drop(columns='quality')
y = train_csv['quality']

train = train_csv.copy()
# for i, v in enumerate(train_csv['quality']):
#     if v < 5:
#         train.loc[i,'quality'] = 0
#     elif v < 7:
#         train.loc[i,'quality'] = 1
#     else:
#         train.loc[i,'quality'] = 2
        
# train['quality'] = (train_csv['quality'] - 2) // 3

g = train_csv.groupby('quality').size()

g_i = g.index
g_v = g.values
print("g_i : ", g_i)
print("g_v : ", g_v)

for i,v  in enumerate(train_csv['quality']):
    if v in g_i[np.where(g_v < 800)]:
        train.loc[i, 'quality'] = 0
    else:
        train.loc[i, 'quality'] = 1
        

g = train.groupby('quality').size()



plt.bar(g.index, g.values)
plt.xlabel('quality')
plt.show()

yy = train['quality']


X_train, X_test, y_train, y_test = train_test_split(X, yy, test_size=0.2, random_state=42, stratify=yy)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = RandomForestClassifier(random_state=42)

model.fit(X_train, y_train, )

y_pred = model.predict(X_test)
f1 = f1_score(y_test, y_pred, average='macro')
print("f1 : ", f1)
print("acc : ", model.score(X_test, y_test))