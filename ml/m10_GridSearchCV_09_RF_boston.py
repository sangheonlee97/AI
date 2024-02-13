import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

datasets= load_boston()
x = datasets.data
y = datasets.target

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, StratifiedKFold, GridSearchCV

x_train, x_test, y_train , y_test = train_test_split(
    x, y, shuffle= True, random_state=123, train_size=0.8,)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=123)

parameters = [
    {"n_estimators": [100, 200], "max_depth": [6, 10, 12], "min_samples_leaf": [3, 10]},
    {"max_depth": [6, 8, 10, 12], "min_samples_leaf": [3, 5, 7, 10]},
    {"min_samples_leaf": [3, 5, 7, 10], "min_samples_split": [2, 3, 5, 10]},
    {"min_samples_split": [2, 3, 5, 10]},
    {"n_jobs": [-1, 2, 4], "min_samples_split": [2, 3, 5, 10]},
]
rfc = RandomForestRegressor()
model = GridSearchCV(rfc, param_grid=parameters, cv=kf , n_jobs=-1, refit=True, verbose=1)
model.fit(x_train, y_train)

from sklearn.metrics import r2_score
best_predict = model.best_estimator_.predict(x_test)
best_acc_score = r2_score(y_test, best_predict)

print("best_model_acc_score : ", best_acc_score) #best_acc_score :  0.9333333333333333

print(f'''
최적의 파라미터 :\t{model.best_estimator_}
최적의 매개변수 :\t{model.best_params_}
best score :\t\t{model.best_score_}
best_model_acc_score :\t{best_acc_score}
''')

'''
최적의 파라미터 :       RandomForestRegressor(n_jobs=-1)
최적의 매개변수 :       {'min_samples_split': 2, 'n_jobs': -1}
best score :            0.8574265827431973
best_model_acc_score :  0.7903848069966082
'''