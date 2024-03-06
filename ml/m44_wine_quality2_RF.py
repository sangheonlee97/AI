from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
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
y = y - 3
print(X.shape, y.shape) # 569, 30
print(np.unique(y, return_counts=True)) # binary ce

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = RandomForestClassifier(random_state=42)

model.fit(X_train, y_train, )


print("model.score : ", model.score(X_test, y_test ))
