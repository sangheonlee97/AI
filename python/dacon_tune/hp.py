from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
from sklearn.model_selection import train_test_split

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
RANDOM_SEED = 42

df = pd.read_csv('../_data/dacon/tune/train.csv', index_col=0)
X = df.drop(columns='login')
y = df['login']
print(X.shape)

# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=.1, random_state=RANDOM_SEED)


def rff(search_space):
    params = {
                'n_estimators': int(search_space['n_estimators']),
                'max_depth': int(search_space['max_depth']),
                'min_samples_split': search_space['min_samples_split'],
                'min_samples_leaf': search_space['min_samples_leaf'],
                'max_features': search_space['max_features'],
                'bootstrap': bool(search_space['bootstrap']),
                'criterion': search_space['criterion'],
                'max_leaf_nodes': int(search_space['max_leaf_nodes']),
                'min_impurity_decrease': search_space['min_impurity_decrease'],
                'min_weight_fraction_leaf': search_space['min_weight_fraction_leaf'],
            }
    
    model = RandomForestClassifier(**params, n_jobs=1)
    acc = cross_val_score(model, X, y, cv=5, scoring='accuracy', n_jobs=1).max()
    
    return -acc

trial_val = Trials()
best = fmin(
    fn=rff,
    space=search_space,
    algo=tpe.suggest,
    max_evals=300,
    trials=trial_val,
    rstate=np.random.default_rng(seed=RANDOM_SEED)
)

print('best : ', best)
print("acc : ", -(trial_val.best_trial['result']['loss']))

submit = pd.read_csv('../_data/dacon/tune/sample_submission.csv')

# 찾은 최적의 파라미터들을 제출 양식에 맞게 제출
for param, value in best.items():
    print(f"{param}: {value} // 1")
    if param in submit.columns:
        if param == 'bootstrap':
            submit[param] = [False, True][value]
        elif param == 'criterion':
            submit[param] = ['gini', 'entropy'][value]
        elif param == 'max_features':
            submit[param] = ['auto', 'sqrt', 'log2'][value]
        elif param == 'min_samples_split':
            submit[param] = int(1309 * value)
        elif param == 'min_samples_leaf':
            submit[param] = int(1309 * value)
        else:
            submit[param] = value
    print(f"{param}: {submit[param][0]} //  2")

submit['max_depth'] = (submit['max_depth']).astype('uint8')
submit['max_leaf_nodes'] = (submit['max_leaf_nodes']).astype('uint8')
submit['n_estimators'] = (submit['n_estimators']).astype('uint8')
print(submit)
submit.to_csv('../_data/dacon/tune/hp.csv', index=False)

pa = {}
pa = submit.iloc[0]
model2 = RandomForestClassifier()
model2.set_params(**pa)
model2.fit(X,y)
print(model2.score(X,y))