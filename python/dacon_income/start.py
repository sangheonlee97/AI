import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split


path = '../_data/dacon/income/'

# data
train_csv = pd.read_csv(path + 'train.csv', index_col=0)    # (20000, 22)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)      # (10000, 21)
sub_csv = pd.read_csv(path + 'sample_submission.csv')

test_csv = test_csv.ffill()     # Household_Status 에 결측치 한개 있음
# for col in train_csv.columns:
#     if train_csv[col].dtype == 'object':
#         print(np.unique(train_csv[col], return_counts=True))
print(np.unique(train_csv['Employment_Status'], return_counts=True))
    
# print((np.unique(train_csv['Birth_Country']) == np.unique(train_csv['Birth_Country (Father)'])))    # 인코딩 할 때, 한개의 인코더 사용 가능 여부 확인
X = train_csv.drop(columns='Income')
y = train_csv['Income']
def find_max_income_for_status(df, status='Under Median'):
    # 특정 조건을 만족하는 행 필터링
    filtered_rows = df[df['Income_Status'] == status]

    # 특정 조건을 만족하는 행이 없는 경우 예외 처리
    if filtered_rows.empty:
        return None

    # 특정 조건을 만족하는 행 중에서 'Income' 컬럼이 가장 큰 값 찾기
    max_income = filtered_rows['Income'].max()
    print(np.unique(filtered_rows['Income'], return_counts=True))

    return max_income

def find_min_income_for_status(df, status='Over Median'):
    # 특정 조건을 만족하는 행 필터링
    filtered_rows = df[df['Income_Status'] == status]

    # 특정 조건을 만족하는 행이 없는 경우 예외 처리
    if filtered_rows.empty:
        return None

    # 특정 조건을 만족하는 행 중에서 'Income' 컬럼이 가장 큰 값 찾기
    min_income = filtered_rows['Income'].min()
    print(np.unique(filtered_rows['Income'], return_counts=True))
    return min_income

education_status = {
    'Children' : 0,
    'Kindergarten' : 1,
    'Elementary (1-4)' : 2,
    'Elementary (5-6)' : 3,
    'Middle (7-8)' : 4,
    'High Freshman' : 5,
    'High Sophomore' : 6,
    'High Junior' : 7,
    'High Senior' : 8,
    'High graduate' : 9,
    'College' : 10,
    'Associates degree (Academic)' : 11,
    'Associates degree (Vocational)' : 12,
    'Bachelors degree' : 13,
    'Masters degree' : 14,
    'Doctorate degree' : 15,
    'Professional degree' : 16
}

gender = {'F' : 0, 'M' : 1}
train_csv['Education_Status'] = train_csv['Education_Status'].map(education_status)
train_csv['Gender'] = train_csv['Gender'].map(gender)

print(train_csv)

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42, )

# model = XGBRegressor()
# model.fit(X_train, y_train)

# print(model.score(X_test, y_test))