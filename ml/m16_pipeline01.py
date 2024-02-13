# from sklearn.datasets import load_iris
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# from sklearn.svm import SVC
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline



# datasets = load_iris()
# X = datasets.data
# y = datasets.target
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y )

# # scaler = MinMaxScaler()
# # X_train = scaler.fit_transform(X_train)
# # X_test = scaler.transform(X_test)
# # model = RandomForestClassifier()

# model = make_pipeline(MinMaxScaler(), RandomForestClassifier(min_samples_split=2, min_samples_leaf=10, random_state=42))



# model.fit(X_train,y_train)

# results = model.score(X_test, y_test)

# print("model : ", model, ", ",'score : ', results)
# y_pred = model.predict(X_test)

# acc = accuracy_score(y_test, y_pred)
# print("model : ", model, ", ","acc : ", acc)
# print("\n")
# ###################################################################################################################################################

# import numpy as np
# from sklearn.datasets import load_breast_cancer
# from sklearn.model_selection import train_test_split, KFold ,cross_val_score, StratifiedKFold, cross_val_predict
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# from sklearn.metrics import accuracy_score
# from sklearn.preprocessing import MinMaxScaler, StandardScaler
# import warnings
# warnings.filterwarnings('ignore')

# # 1. 데이터
# x,y = load_breast_cancer(return_X_y=True)
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, StratifiedKFold, RandomizedSearchCV

# x_train, x_test, y_train , y_test = train_test_split(
#     x, y, shuffle= True, random_state=123, train_size=0.8,
#     stratify= y
# )


# n_splits = 5
# kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)

# parameters = [
#     {"n_estimators": [100, 200], "max_depth": [6, 10, 12], "min_samples_leaf": [3, 10]},
#     {"max_depth": [6, 8, 10, 12], "min_samples_leaf": [3, 5, 7, 10]},
#     {"min_samples_leaf": [3, 5, 7, 10], "min_samples_split": [2, 3, 5, 10]},
#     {"min_samples_split": [2, 3, 5, 10]},
#     {"n_jobs": [-1, 2, 4], "min_samples_split": [2, 3, 5, 10]},
# ]
# rfc = RandomForestClassifier()
# model = RandomizedSearchCV(rfc,  parameters, cv=kf , n_jobs=-1, refit=True, verbose=1)
# pipeline = make_pipeline(MinMaxScaler(), model)

# pipeline.fit(x_train, y_train)

# from sklearn.metrics import accuracy_score
# # best_predict = pipeline.best_estimator_.predict(x_test)
# # best_acc_score = accuracy_score(y_test, best_predict)

# print("best_model_acc_score : ", pipeline.score(x_test, y_test)) #best_acc_score :  0.9333333333333333


# '''
# 최적의 파라미터 :       RandomForestClassifier(max_depth=12, min_samples_leaf=3, n_estimators=200)
# 최적의 매개변수 :       {'max_depth': 12, 'min_samples_leaf': 3, 'n_estimators': 200}
# best score :            0.9670329670329672
# best_model_acc_score :  0.9649122807017544
# '''

# # 최적의 파라미터 :       RandomForestClassifier(n_jobs=2)
# # 최적의 매개변수 :       {'n_jobs': 2, 'min_samples_split': 2}
# # best score :            0.9626373626373628
# # best_model_acc_score :  0.9824561403508771
# ###################################################################################################################################################
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_diabetes
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold, cross_val_predict
from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings('ignore')

# 1. 데이터
x, y = load_diabetes(return_X_y=True)

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, StratifiedKFold, RandomizedSearchCV

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
model = RandomizedSearchCV(rfc,  parameters, cv=kf , n_jobs=-1, refit=True, verbose=1)

model.fit(x_train, y_train)
pipeline = make_pipeline(MinMaxScaler(), model )
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
최적의 파라미터 :       RandomForestRegressor(min_samples_leaf=10)
최적의 매개변수 :       {'min_samples_leaf': 10, 'min_samples_split': 2}
best score :            0.41358555127487423
best_model_acc_score :  0.5665789288865843
'''