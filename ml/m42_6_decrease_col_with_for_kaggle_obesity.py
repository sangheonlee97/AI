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
max = { 'col' : 100, 'acc' : 0}
while X.shape[1] >= 1:
    print("X.shape : ", X.shape)
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

    idx = np.argmin(model.feature_importances_)
    print("\n")
    if max["acc"] < acc:
        max['acc'] = acc
        max['col'] = X.shape[1]
    X = np.delete(X, idx, axis=1)
print("최대 acc : ", max['acc'])
print("그때 col 개수 : ", max['col'])