
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV


search_space = {
    'n_estimators': hp.quniform('n_estimators', 100, 500, 10),
    'max_depth': hp.quniform('max_depth', 5, 25, 1),
    'min_samples_split': hp.uniform('min_samples_split', 0.1, 0.5),
    'min_samples_leaf': hp.uniform('min_samples_leaf', 0.1, 0.5),
    'max_features': hp.choice('max_features', [None, 'sqrt', 'log2']),
    'bootstrap': hp.choice('bootstrap', [False, True]),
    'criterion': hp.choice('criterion', ['gini', 'entropy']),
    'max_leaf_nodes': hp.quniform('max_leaf_nodes', 10, 50, 1),
    'min_impurity_decrease': hp.uniform('min_impurity_decrease', 0.0, 0.2),
    'min_weight_fraction_leaf': hp.uniform('min_weight_fraction_leaf', 0.0, 0.2),
}

param_distributions = {
    'n_estimators': list(range(100, 1001, 100)),
    'max_depth': list(range(5, 26, 2)),
    'min_samples_split': list(range(100, 1101, 200)),
    'min_samples_leaf': list(range(100, 1101, 200)),
    'max_features': [None],
    'bootstrap': [True],
    'criterion': ['gini', 'entropy'],
    'max_leaf_nodes': list(range(10, 51,10)),
    'min_impurity_decrease': np.linspace(0.0, 0.2, 5),
    'min_weight_fraction_leaf': np.linspace(0.0, 0.2, 5),
}
RANDOM_SEED = 42

df = pd.read_csv('../_data/dacon/tune/train.csv', index_col=0)
X = df.drop(columns='login')
y = df['login']
print(X.shape)

# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=.1, random_state=RANDOM_SEED)



# RandomForestClassifier 객체 생성
rf = RandomForestClassifier(random_state=42)

# GridSearchCV 객체 생성
# grid_search = RandomizedSearchCV(estimator=rf, param_distributions=param_distributions, cv=5, n_jobs=-5, verbose=2, scoring='roc_auc', n_iter=50)
grid_search = GridSearchCV(estimator=rf, param_grid=param_distributions, cv=5, n_jobs=10, verbose=2, scoring='roc_auc', )

# GridSearchCV를 사용한 학습
grid_search.fit(X, y)

# 최적의 파라미터와 최고 점수 출력
best_params = grid_search.best_params_
best_score = grid_search.best_score_

best_params, best_score

submit = pd.read_csv('../_data/dacon/tune/sample_submission.csv')

# 찾은 최적의 파라미터들을 제출 양식에 맞게 제출
for param, value in best_params.items():
    if param in submit.columns:
        submit[param] = value

submit.to_csv('../_data/dacon/tune/rs.csv', index=False)