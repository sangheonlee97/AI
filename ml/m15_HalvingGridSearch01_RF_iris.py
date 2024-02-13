'''
1. factor 조절
2. min_resources 조절
3. 훈련 데이터를 3이터 이상ㅗ
'''
# 모델 : RanadomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import MinMaxScaler

# 1. 데이터
x,y = load_iris(return_X_y=True)

from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, StratifiedKFold, GridSearchCV, RandomizedSearchCV, HalvingGridSearchCV

x_train, x_test, y_train , y_test = train_test_split(
    x, y, shuffle= True, random_state=123, train_size=0.8,
    stratify= y
)
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_splits = 5
kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)

parameters = [
    {"n_estimators": [100, 200], "max_depth": [6, 10, 12], "min_samples_leaf": [3, 10]},
    {"max_depth": [6, 8, 10, 12], "min_samples_leaf": [3, 5, 7, 10]},
    {"min_samples_leaf": [3, 5, 7, 10], "min_samples_split": [2, 3, 5, 10]},
    {"min_samples_split": [2, 3, 5, 10]},
    {"n_jobs": [-1, 2, 4], "min_samples_split": [2, 3, 5, 10]},
]
rfc = RandomForestClassifier()
model = HalvingGridSearchCV(rfc, parameters, cv=kf , n_jobs=-1, refit=True, verbose=1, factor=2, min_resources=30)
model.fit(x_train, y_train)

from sklearn.metrics import accuracy_score
best_predict = model.best_estimator_.predict(x_test)
best_acc_score = accuracy_score(y_test, best_predict)

print("best_model_acc_score : ", best_acc_score) #best_acc_score :  0.9333333333333333

print(f'''
최적의 파라미터 :\t{model.best_estimator_}
최적의 매개변수 :\t{model.best_params_}
best score :\t\t{model.best_score_}
best_model_acc_score :\t{best_acc_score}
''')

'''
최적의 파라미터 :       RandomForestClassifier(max_depth=6, min_samples_leaf=3)
최적의 매개변수 :       {'max_depth': 6, 'min_samples_leaf': 3, 'n_estimators': 100}
best score :            0.9583333333333334
best_model_acc_score :  0.9333333333333333
'''

# 최적의 파라미터 :       RandomForestClassifier(min_samples_leaf=10)
# 최적의 매개변수 :       {'min_samples_split': 2, 'min_samples_leaf': 10}
# best score :            0.9583333333333334
# best_model_acc_score :  0.9666666666666667