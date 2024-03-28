import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import *
from sklearn.metrics import mean_squared_error

path = "../_data/dacon/income/"

train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission_csv = pd.read_csv(path + "sample_submission.csv")

new_train_csv = train_csv.copy()
new_train_csv = new_train_csv[new_train_csv['Income'] != 0]

X = new_train_csv.drop(['Income'], axis=1)
y = new_train_csv['Income']
test = test_csv
lb = LabelEncoder()

# 라벨 인코딩할 열 목록
columns_to_encode = ['Gender','Education_Status','Employment_Status','Industry_Status','Occupation_Status','Race','Hispanic_Origin','Martial_Status','Household_Status','Household_Summary','Citizenship','Birth_Country','Birth_Country (Father)','Birth_Country (Mother)','Tax_Status','Income_Status']

# 데이터프레임 x의 열에 대해 라벨 인코딩 수행
for column in columns_to_encode:
    lb.fit(X[column])
    X[column] = lb.transform(X[column])

# 데이터프레임 test_csv의 열에 대해 라벨 인코딩 수행
for column in columns_to_encode:
    lb.fit(test_csv[column])
    test_csv[column] = lb.transform(test_csv[column])
    
# 데이터 스케일링
scaler = StandardScaler()
# scaler = MinMaxScaler()

X = scaler.fit_transform(X)
test_csv = scaler.transform(test_csv)

import random

r = random.randint(1,500)

# 훈련 데이터와 검증 데이터 분리
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=107533423)

# XGBoost 모델 학습
xgb_params = {'learning_rate': 0.008,
            'n_estimators': 300,
            'max_depth': 15,
            'min_child_weight': 0.07709868781803283,
            'subsample': 0.70309973945344,
            'colsample_bytree': 0.8254025887963853,
            'gamma': 6.628562492458777e-08,
            'reg_alpha': 0.012998871754325427,
            'reg_lambda': 0.10637051171111844}

model = xgb.XGBRegressor(**xgb_params)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=30, verbose=100)
import joblib
save_path = ''.join(path + "income_xgb_03_28_test3")
# 모델 저장
joblib.dump(model, save_path + ".pkl")

# 저장된 모델 불러오기
# loaded_model = joblib.load("../_data/dacon/income/weights/money_xgb_03_18_2.pkl")
# 검증 데이터 예측
y_pred_val = model.predict(X_val)

# 검증 데이터 RMSE 계산
rmse_val = mean_squared_error(y_val, y_pred_val, squared=False)
print("Validation RMSE:", rmse_val,'r',r)

y_submit = model.predict(test_csv)  
# y_submit = lae.inverse_transform(y_submit)
# y_submit = lae.inverse_transform(y_submit)
submission_csv['Income'] = y_submit
submission_csv.loc[submission_csv['Income'] < 200, 'Income'] = 0.0
print(y_submit)

submission_csv.to_csv(save_path + ".csv", index=False)