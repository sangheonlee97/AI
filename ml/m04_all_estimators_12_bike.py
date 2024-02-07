import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score, mean_squared_log_error
from keras.callbacks import EarlyStopping,ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler

path = "..\\_data\\kaggle\\bike\\"

train_csv = pd.read_csv(path + "train.csv" , index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission_csv = pd.read_csv(path + "samplesubmission.csv")

X = train_csv.drop(['casual', 'registered', 'count'], axis=1) 
y = train_csv['count']

# print(X.shape)      #(10886, 8)
# print(y.shape)      #(10886)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=713)

#############    MinMaxScaler    ##############################
mms = MinMaxScaler()
mms.fit(X_train)
X_train = mms.transform(X_train)
X_test = mms.transform(X_test)
test_csv = mms.transform(test_csv)

################    StandardScaler    ##############################

# sts = StandardScaler()
# sts.fit(X_train)
# X_train = sts.transform(X_train)
# X_test = sts.transform(X_test)

# print(X_train)
# print(X_test)

# ################    MaxAbsScaler    ##############################
# mas = MaxAbsScaler()
# mas.fit(X_train)
# X_train = mas.transform(X_train)
# X_test = mas.transform(X_test)


# ################    RobustScaler    ##############################
# rbs = RobustScaler()
# rbs.fit(X_train)
# X_train = rbs.transform(X_train)
# X_test = rbs.transform(X_test)

# model = Sequential()
# model.add(Dense(19, input_shape= (8, ),activation='relu'))
# model.add(Dense(97))
# model.add(Dense(9,activation='relu'))
# model.add(Dense(21))
# model.add(Dense(16,activation='relu'))
# model.add(Dense(21))
# model.add(Dense(1, activation='relu'))
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

    # # MinMaxScaler
    # RMSLE :  4.800237655708398

    # # MaxAbsScaler
    # RMSLE :  1.305517434934688
    # # StandardScaler
    # RMSLE :  1.3162207010884128

    # # RobustScaler
    # RMSLE :  1.2885431257772446



    # cpu : 40.75
    # gpu : 73.91