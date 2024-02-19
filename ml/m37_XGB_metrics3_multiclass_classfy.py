from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, GridSearchCV, RandomizedSearchCV
# from sklearn.model_selection import HalvingGridSearchCV, HalvingRandomSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import f1_score, accuracy_score
import numpy as np
import pandas as pd
n_splits = 5
param = {
            'n_estimators' : 5000,                # default 100 / 1~inf / 정수
            'learning_rate' : 0.001,           # default 0.3 / 0~1 / eta
            'max_depth' :  10 ,                 # default 6 / 0~inf / 정수
            # 'gamma' : [0,1,2,3,4,5,7,10, 100] ,                               # default 0 / 0~inf
            # 'min_child_weight' : [0, 0.01, 0.001, 0.1, 0.5, 1, 5, 10, 100],   # default 1 / 0~inf
            # 'subsample' : [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1],                    # default 1 / 0~1
            # 'colsample_bytree' : [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1],             # default 1 / 0~1
            # 'colsample_bylevel' : [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1],            # default 1 / 0~1
            # 'colsample_bynode' : [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1],             # default 1 / 0~1
            # 'reg_alpha' : [0, 0.1, 0.01, 0.001, 1, 2, 10],                    # default 0 / 0~inf / L1 절대값 가중치 규제 / alpha
            # 'reg_lambda' : [0, 0.1, 0.01, 0.001, 1, 2, 10],                   # default 1 / 0~inf / L2 제곱 가중치 규제 / lambda        
}
# 파라미터 튜닝 맛집 : n_estimator( epochs 같은 느낌. 클수록 좋다. ), learning_rate( 작으면 작을수록 촘촘하게 학습. 작을수록 성능이 좋다. ), 

X, y = load_digits(return_X_y=True)
print(X.shape, y.shape) # 569, 30
# print(np.unique(y, return_counts=True)) # binary ce

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

xgb = XGBClassifier(random_state=42, )
xgb.set_params(**param, early_stopping_rounds=30)
# xgb.set_params(**param)

xgb.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=1, 
        # eval_metric='mse' # error
        # eval_metric='logloss' #  이진분류 default
        # eval_metric='mlogloss' #  다중분류 default
        # eval_metric='merror' # 다중분류
        # eval_metric='error' # 이진분류
        # eval_metric='mae' # 회귀
        # eval_metric='rmse' # 회귀 default
        # eval_metric='auc' # 이진, 다중 다 ( 하지만 이진이 더 좋다) 
        )
print(xgb.get_params())
print("model.score : ", xgb.score(X_test, y_test))
y_pred = xgb.predict(X_test)
f1 = f1_score(y_test, y_pred, average='macro')
print("f1 : ", f1)

acc = accuracy_score(y_test, y_pred)
print("acc : ", acc)

# model.score :  0.9583333333333334
# f1 :  0.9582341019004555
# acc :  0.9583333333333334