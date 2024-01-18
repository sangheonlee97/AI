
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.callbacks import EarlyStopping,ModelCheckpoint
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler


path = "..\\_data\\dacon\\wine\\"

train_csv = pd.read_csv(path + "train.csv", index_col=0)       
# train_csv.to_csv(path + "train_123_csv", index=False)                                             
test_csv = pd.read_csv(path + "test.csv", index_col=0)

submission_csv = pd.read_csv(path + "sample_submission.csv")

# print(train_csv)
# print(train_csv.shape)      #(5497, 13)
# print(test_csv)
# print(test_csv.shape)       #(1000, 12)


# print(X.shape)      #(5497, 12)
# print(y)
# print(y.shape)      #(5497, )




     
lae = LabelEncoder()
lae.fit(train_csv['type'])
train_csv['type'] = lae.transform(train_csv['type'])
test_csv['type'] = lae.transform(test_csv['type'])

X = train_csv.drop(['quality'], axis=1)
# print(X)
# print(X.shape)
y = train_csv['quality']
# print(y.shape)

# mms = MinMaxScaler
# mms.fit(X)
# X = mms.transform(X)
# test_csv = mms.transform(test_csv)

y = pd.get_dummies(y)

# print(y)
# print(y.shape)      #(5497, 7)      # 3,4,5,6,7,8,9

# print(X)

# print(X.shape)          # (5497, 12)
# print(y.shape)          #(5497, 7)
# print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.325, shuffle=True, random_state=3, stratify=y)       #9266, 781

##############    MinMaxScaler    ##############################
# mms = MinMaxScaler()
# mms.fit(X_train)
# X_train = mms.transform(X_train)
# X_test = mms.transform(X_test)
# test_csv = mms.transform(test_csv)
################    StandardScaler    ##############################

# sts = StandardScaler()
# sts.fit(X_train)
# X_train = sts.transform(X_train)
# X_test = sts.transform(X_test)
# test_csv = sts.transform(test_csv)

# print(X_train)
# print(X_test)

# ################    MaxAbsScaler    ##############################
# mas = MaxAbsScaler()
# mas.fit(X_train)
# X_train = mas.transform(X_train)
# X_test = mas.transform(X_test)


# ################    RobustScaler    ##############################
rbs = RobustScaler()
rbs.fit(X_train)
X_train = rbs.transform(X_train)
X_test = rbs.transform(X_test)
test_csv = rbs.transform(test_csv)
# X_test = rbs.transform(X_test)

# model = Sequential()
# model.add(Dense(19, input_dim=12,activation='relu'))
# model.add(Dense(97,activation='relu'))             
# model.add(Dense(9))      
# model.add(Dense(21,activation='relu'))           
# model.add(Dense(16))
# model.add(Dense(21,activation='relu'))      
# model.add(Dense(7, activation='softmax'))
from keras.models import Model
from keras.layers import Input
inp = Input(shape=(12, ))
d1 = Dense(19, activation='relu')(inp)
d2 = Dense(97)(d1)
d3 = Dense(9, activation='relu')(d2)
d4 = Dense(21)(d3)
op = Dense(7, activation='softmax')(d4)
model = Model(inputs=inp, outputs=op)



import datetime
date = datetime.datetime.now()
# print(date)                     # 2024-01-17 10:52:59.857389
date = date.strftime("%m%d_%H%M")                   # _는 str      
# print(date)                     # 0117_1058
# print(type(date))               # <class 'str'>

path = "..\\_data\\_save\\MCP\\"
filename = '{epoch:05d}-{acc:.4f}-{loss:.4f}.hdf5'            # 04d : 4자리 정수표현, 4f : 소수4번째자리까지 표현, 예) 1000_0.3333.hdf5
filepath = "".join([path, 'k26_10_dacon_wine_',date,'_', filename])

mcp = ModelCheckpoint(monitor='val_loss', mode='min', verbose=1, save_best_only=True, filepath=filepath)    
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')
es = EarlyStopping(monitor='val_loss', mode='min', patience=300, verbose=3, restore_best_weights=True)
model.fit(X_train, y_train, epochs=2000, batch_size=270, validation_split=0.125, callbacks=[es,mcp], verbose=2)





results = model.evaluate(X_test, y_test)
print("ACC : ", results[1])


# print(X_test)
# print(X_train)

# print(test_csv)
y_submit = model.predict(test_csv)  
y_predict = model.predict(X_test) 

y_test = np.argmax(y_test, axis=1)
y_predict = np.argmax(y_predict, axis=1)
y_submit = np.argmax(y_submit, axis=1)+3

# print(y_test)
# print(y_predict)
# print(y_submit)

submission_csv['quality'] =y_submit
# # print(y_test)
# # print(y_predict)
# print(y_submit)
# print(y_submit.shape) 


submission_csv.to_csv(path + "submission_0116_11_.csv", index=False)
print("로스 : ", results[0])

acc = accuracy_score(y_predict, y_test)
print("accuracy_score : ", acc)
print("로스 : ", results[0])
print("ACC : ", results[1])

# 로스 :  1.0687559843063354
# accuracy_score :  0.552069122328331
# 로스 :  1.0687559843063354
# ACC :  0.5520691275596619




# MinMaxScaler
# 로스 :  1.0641732215881348
# accuracy_score :  0.5402455661664393
# 로스 :  1.0641732215881348
# ACC :  0.5402455925941467

# MaxAbsScaler
# 로스 :  1.0634896755218506
# accuracy_score :  0.5447930877671668
# 로스 :  1.0634896755218506
# ACC :  0.5447930693626404

# StandardScaler
# 로스 :  1.0636837482452393
# accuracy_score :  0.5493406093678945
# 로스 :  1.0636837482452393
# ACC :  0.5493406057357788

# RobustScaler
# 로스 :  1.0621486902236938
# accuracy_score :  0.5461573442473852
# 로스 :  1.0621486902236938
# ACC :  0.5461573600769043





