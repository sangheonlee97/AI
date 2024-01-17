import numpy as np
import pandas as pd
from keras.models import Sequential, load_model
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score, mean_squared_log_error
from keras.callbacks import EarlyStopping,ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler

path = "c:\\_data\\kaggle\\bike\\"

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

# mcp = ModelCheckpoint(monitor='val_loss', mode='min', verbose=1, save_best_only=True, filepath="c:\\_data\\_save\\MCP\\keras26_MCP05_kaggle_bike.hdf5")    
# model.compile(loss='mse', optimizer='adam', metrics='accuracy')
# es = EarlyStopping(monitor='val_loss', mode='min', patience=300, restore_best_weights=True)
# hist = model.fit(X_train, y_train, epochs= 1500, batch_size=700, validation_split=0.15,callbacks=[es,mcp])


model = load_model("..\\_data\\_save\\MCP\\k26_5_kaggle_bike_0117_1404_01469-22035.9316-21063.4277.hdf5")


loss = model.evaluate(X_test, y_test)
y_submit = model.predict(test_csv)
y_predict = model.predict(X_test)
submission_csv['count'] = y_submit
print("mse : ",loss )
submission_csv.to_csv(path + "submission_0116_2.csv", index=False)

print("음수 : ", submission_csv[submission_csv['count']<0].count())

r2 = r2_score(y_test, y_predict)
def RMSLE(y_test, y_predict):
    np.sqrt(mean_squared_log_error(y_test, y_predict))
    return np.sqrt(mean_squared_log_error(y_test, y_predict))
rmsle = RMSLE(y_test, y_predict) 

print("RMSLE : ", rmsle)


# MinMaxScaler
RMSLE :  4.800237655708398

# MaxAbsScaler
RMSLE :  1.305517434934688
# StandardScaler
RMSLE :  1.3162207010884128

# RobustScaler
RMSLE :  1.2885431257772446


