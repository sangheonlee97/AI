import warnings
warnings.filterwarnings('ignore')
from sklearn.datasets import load_boston
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler
# 현재 사이킷런 버전 1.3.0이라 보스턴 안됨. 그래서 삭제
# pip uninstall scikit-learn
# pip uninstall scikit-learn-intelex
# pip uninstall scikit-image
# pip install scikit-learn==1.1.3
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
import numpy as np

datasets = load_boston()
print(datasets)
X = datasets.data
y = datasets.target
# print(X.shape)  # (506, 13)
# print(y.shape)  # (506, )

# print(datasets.feature_names)   # ['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO' 'B' 'LSTAT']

# print(datasets.DESCR)   # 속성

# [실습]
# train_size 0.7이상 0.9 이하
# R2 0.62 이상

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=587)

mms = MinMaxScaler()
mms.fit(X_train)
X_train = mms.transform(X_train)
X_test = mms.transform(X_test)

# model = Sequential()
# model.add(Dense(25, input_dim=13))
# model.add(Dense(40))
# model.add(Dense(35))
# model.add(Dense(30))
# model.add(Dense(1))
from sklearn.utils import all_estimators
from sklearn.metrics import r2_score
allAlgorithms = all_estimators(type_filter='regressor')
best = [0, 'no']
for name, algorithm in allAlgorithms:
    try:
        model = algorithm()
        
        model.fit(X_train,y_train)

        # results = model.score(X_test, y_test)

        # print("model : ", model, ", ",'score : ', results)
        y_pred = model.predict(X_test)

        acc = r2_score(y_test, y_pred)
        print("model : ", name, ", ","r2 : ", acc)
        if best[0] < acc:
            best[0] = acc
            best[1] = name
    except:
        print("qudtlsrkxdms dkfrhflwma : ", name)
print("best model : ", best[1], "\nbest r2 : ", best[0])