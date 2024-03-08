import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from bayes_opt import BayesianOptimization
import warnings
warnings.filterwarnings('ignore')
X, y = load_breast_cancer(return_X_y=True)
bayesian_params = {
            'learning_rate' : (0.001, 1),          # default 0.3 / 0~1 / eta
            'max_depth' :  (3, 10) ,                 # default 6 / 0~inf / 정수
            'min_child_samples' : (10, 200),   # default 1 / 0~inf
            'min_child_weight' : (1, 50),   # default 1 / 0~inf
            'subsample' : (0.5 , 1),                  # default 1 / 0~1
            'colsample_bytree' : (0.5, 1),             # default 1 / 0~1
            'reg_alpha' : (0.01 , 50),                    # default 0 / 0~inf / L1 절대값 가중치 규제 / alpha
            'reg_lambda' : (-0.001, 10) ,                  # default 1 / 0~inf / L2 제곱 가중치 규제 / lambda        
            'max_bin' : (9, 500),
            'num_leaves' : (24, 40)
}

def xgb_hamsu(learning_rate, max_depth, num_leaves, min_child_samples, min_child_weight, subsample, colsample_bytree, max_bin, reg_lambda, reg_alpha):
    params = { 
              'n_estimators' : 100,
              'learning_rate' : learning_rate,
              'max_depth' : int(round(max_depth)),
              'num_leaves' : int(round(num_leaves)),
              'min_child_samples' : int(round(min_child_samples)),
              'min_child_weight' : int(round(min_child_weight)),
              'subsample' : max(min(subsample, 1), 0),
              'colsample_bytree' : colsample_bytree,
              'max_bin' : max(int(round(max_bin)), 10),
              'reg_lambda' : max(reg_lambda, 0),
              'reg_alpha' : reg_alpha,
              }
    
    model = XGBClassifier(**params, n_jobs=-3)
    
    model.fit(X_train, y_train,
              eval_set=[(X_train, y_train), (X_test, y_test)],
              eval_metric='logloss',
              verbose=0,
              early_stopping_rounds=50,
    )
    y_predict = model.predict(X_test)
    results = accuracy_score(y_test, y_predict)
    return results

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=777, stratify=y)

ms = MinMaxScaler()
X_train = ms.fit_transform(X_train)
X_test = ms.transform(X_test)
import time
start_time = time.time()
bay = BayesianOptimization(f=xgb_hamsu,
                           pbounds=bayesian_params,
                           random_state=42)

n_iter = 100
bay.maximize(init_points=5, n_iter=n_iter)
end_time = time.time()
# print(bay.max)
print(n_iter, '번 걸린 시간 : ', round(end_time - start_time), '초')