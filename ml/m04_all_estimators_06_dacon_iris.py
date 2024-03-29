import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
# 클론 테스트
path = "..\\_data\\dacon\\iris\\"

train_csv = pd.read_csv(path + "train.csv", index_col='id')
# print(train_csv.shape)  # (120, 5)

test_csv = pd.read_csv(path + "test.csv", index_col='id')
# print(test_csv.shape)   # (30, 4)

submission_csv = pd.read_csv( path + "sample_submission.csv")
# print(submission_csv.shape) # (30, 2)

print(train_csv)

X = train_csv.iloc[:,:-1]
y = train_csv.iloc[:,-1]

# print(X.shape)  # (120, 4)
# print(y.shape)  # (120,)
# print(y.value_counts())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)



from sklearn.metrics import accuracy_score
from sklearn.utils import all_estimators
allAlgorithms = all_estimators(type_filter='classifier')
best = [0, 'no', 9999]
for i, v in enumerate(allAlgorithms):
    name, algorithm = v
    try:
        model = algorithm()
        
        model.fit(X_train,y_train)

        # results = model.score(X_test, y_test)

        # print("model : ", model, ", ",'score : ', results)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        print("idx : ", i, ", model : ", name, ", ","acc : ", acc)
        if best[0] < acc:
            best[0] = acc
            best[1] = name
            best[2] = i
    except:
        print("qudtlsrkxdms dkfrhflwma : ", name)
print("best model : ", best[1], ", idx [", best[2],"]", "\nbest acc : ", best[0])

# best model :  AdaBoostClassifier , idx [ 0 ]
# best acc :  0.9791666666666666