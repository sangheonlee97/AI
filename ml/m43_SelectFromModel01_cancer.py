import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, r2_score
parameter = {
            'n_estimators' : 1000,                # default 100 / 1~inf / 정수
            'learning_rate' : 0.01,           # default 0.3 / 0~1 / eta
            'max_depth' :  3 ,                 # default 6 / 0~inf / 정수
            'gamma' : 0 ,                               # default 0 / 0~inf
            'min_child_weight' : 0,   # default 1 / 0~inf
            'subsample' : 0.4,                  # default 1 / 0~1
            'colsample_bytree' : 0.8,             # default 1 / 0~1
            'colsample_bylevel' : 0.7,            # default 1 / 0~1
            'colsample_bynode' : 1 ,             # default 1 / 0~1
            'reg_alpha' : 0,                    # default 0 / 0~inf / L1 절대값 가중치 규제 / alpha
            'reg_lambda' : 1 ,                  # default 1 / 0~inf / L2 제곱 가중치 규제 / lambda        
            'random_state' : 3377,
}
df = load_breast_cancer()
X = df.data
y = df.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, )

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = XGBClassifier()
model.set_params(early_stopping_rounds=10, **parameter,eval_metric='logloss')

model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=0, )

res = model.score(X_test, y_test)
print("최종 점수 : ", res)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("acc : ", acc)

thresholds = np.sort(model.feature_importances_)

from sklearn.feature_selection import SelectFromModel

for i in thresholds:
    selection = SelectFromModel(model, threshold=i, prefit=True)
    
    select_X_train = selection.transform(X_train)
    select_X_test = selection.transform(X_test)
    print("변형된 X : ", select_X_train.shape)
    select_model = XGBClassifier()
    select_model.set_params(early_stopping_rounds=10, **parameter, eval_metric='logloss')
    
    select_model.fit(select_X_train, y_train, eval_set=[(select_X_train, y_train), (select_X_test, y_test)], verbose=0, )
    
    result = select_model.score(select_X_test, y_test)
    print("result : ", result)
    