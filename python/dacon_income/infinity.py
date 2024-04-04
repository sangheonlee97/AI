import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
import joblib
import os
import random
import json

path = "../_data/dacon/income/"

train_csv = pd.read_csv(os.path.join(path, "train.csv"), index_col=0)
test_csv = pd.read_csv(os.path.join(path, "test.csv"), index_col=0)
submission_csv = pd.read_csv(os.path.join(path, "sample_submission.csv"))

X = train_csv.drop(['Income'], axis=1)
y = train_csv['Income']
test = test_csv

lb = LabelEncoder()

# 라벨 인코딩할 열 목록
columns_to_encode = ['Gender','Education_Status','Employment_Status','Industry_Status','Occupation_Status','Race','Hispanic_Origin','Martial_Status','Household_Status','Household_Summary','Citizenship','Birth_Country','Birth_Country (Father)','Birth_Country (Mother)','Tax_Status','Income_Status']

# 데이터프레임 X의 열에 대해 라벨 인코딩 수행
for column in columns_to_encode:
    lb.fit(X[column])
    X[column] = lb.transform(X[column])

# 데이터프레임 test_csv의 열에 대해 라벨 인코딩 수행
for column in columns_to_encode:
    lb.fit(test_csv[column])
    test_csv[column] = lb.transform(test_csv[column])

# 데이터 스케일링
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
test_scaled = scaler.transform(test_csv)

# 훈련 데이터와 검증 데이터 분리
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
min = 999999
# 무한 루프 시작
while True:
    # 랜덤 파라미터 생성
    xgb_params = {'learning_rate': random.uniform(0.0001, 0.01),
                'n_estimators': random.randint(100, 500),
                'max_depth': random.randint(5, 25),
                'min_child_weight': random.uniform(0.05, 0.1),
                'subsample': random.uniform(0.6, 0.8),
                'colsample_bytree': random.uniform(0.7, 0.9),
                'gamma': random.uniform(0, 0.1),
                'reg_alpha': random.uniform(0.01, 0.02),
                'reg_lambda': random.uniform(0.1, 0.2)
                }

    # XGBoost 모델 학습
    model = xgb.XGBRegressor(**xgb_params, n_jobs=16)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=35, verbose=100)

    y_pred_val = model.predict(X_val)
    rmse_val = mean_squared_error(y_val, y_pred_val, squared=False)
    print("Validation RMSE:", rmse_val)

    if not os.path.exists(path):
        os.makedirs(path)

    if min > rmse_val:
        # 모델 저장
        save_path_model = os.path.join(path, f"income_xgb_04_03_{rmse_val:.2f}.pkl")
        joblib.dump(model, save_path_model)

        # 파라미터 저장
        save_path_params = os.path.join(path, f"income_xgb_04_03_{rmse_val:.2f}.json")
        with open(save_path_params, 'w') as f:
            json.dump(xgb_params, f)

        # submission 파일 저장
        save_path_csv = os.path.join(path, f"income_xgb_04_03_{rmse_val:.2f}.csv")
        submission_csv['Income'] = model.predict(test_scaled)
        submission_csv.to_csv(save_path_csv, index=False)
        min = rmse_val
