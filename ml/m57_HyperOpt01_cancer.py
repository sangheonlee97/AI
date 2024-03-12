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
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
import warnings
warnings.filterwarnings('ignore')
X, y = load_breast_cancer(return_X_y=True)
search_space = {
            'learning_rate' : hp.uniform('learning_rate', 0.001, 1),          # default 0.3 / 0~1 / eta
            'max_depth' :  hp.quniform('max_depth', 3, 10, 1) ,                 # default 6 / 0~inf / 정수
            'min_child_samples' : hp.quniform('min_child_samples', 10, 200, 1),   # default 1 / 0~inf
            'min_child_weight' : hp.quniform('min_child_weight', 1, 50, 1),   # default 1 / 0~inf
            'subsample' : hp.uniform('subsample', 0.5 , 1),                  # default 1 / 0~1
            'colsample_bytree' : hp.uniform('colsample_bytree', 0.5, 1),             # default 1 / 0~1
            'reg_alpha' : hp.uniform('reg_alpha', 0.01 , 50),                    # default 0 / 0~inf / L1 절대값 가중치 규제 / alpha
            'reg_lambda' : hp.uniform('reg_lambda', -0.001, 10) ,                  # default 1 / 0~inf / L2 제곱 가중치 규제 / lambda        
            'max_bin' : hp.quniform('max_bin', 9, 500, 1),
            'num_leaves' : hp.quniform('num_leaves', 24, 40, 1)
}

def xgb_hamsu(search_space):
    params = { 
              'n_estimators' : 100,
              'learning_rate' : search_space['learning_rate'],
              'max_depth' : int(search_space['max_depth']),
              'num_leaves' : int(search_space['num_leaves']),
              'min_child_samples' : int(search_space['min_child_samples']),
              'min_child_weight' : int(search_space['min_child_weight']),
              'subsample' : max(min(search_space['subsample'], 1), 0),
              'colsample_bytree' : search_space['colsample_bytree'],
              'max_bin' : max(int(round(search_space['max_bin'])), 10),
              'reg_lambda' : max(search_space['reg_lambda'], 0),
              'reg_alpha' : search_space['reg_alpha'],
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
    
    return -results

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=777, stratify=y)

ms = MinMaxScaler()
X_train = ms.fit_transform(X_train)
X_test = ms.transform(X_test)
import time
start_time = time.time()

trial_val = Trials()
best = fmin(
    fn=xgb_hamsu,
    space=search_space,
    algo=tpe.suggest,
    max_evals=50,
    trials=trial_val,
    rstate=np.random.default_rng(seed=10)
)


print('best : ', best)
end_time = time.time()
print('걸린 시간 : ', round(end_time - start_time), '초')
print("acc : ", -(trial_val.best_trial['result']['loss']))
# best :  { 'colsample_bytree': 0.5300418778765242, 
#           'learning_rate': 0.12672595272780718, 
#           'max_bin': 313.0, 
#           'max_depth': 6.0, 
#           'min_child_samples': 135.0, 
#           'min_child_weight': 4.0, 
#           'num_leaves': 34.0, 
#           'reg_alpha': 0.48385618387130336, 
#           'reg_lambda': 9.913429190547106, 
#           'subsample': 0.9276299549911071}
# 걸린 시간 :  2 초
# acc :  0.9912280701754386
