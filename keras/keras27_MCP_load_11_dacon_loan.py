import numpy as np
import pandas as pd
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.callbacks import EarlyStopping,ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler, RobustScaler
from keras.utils import to_categorical


# 1. 데이터
path = "c:\\_data\\dacon\\loan\\"

train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission_csv = pd.read_csv(path + "sample_submission.csv")

# print(train_csv)            #(96234, 14)
# print(test_csv)             #(64197, 13)
# print(submission_csv)       #(6497, 2)
# print(X.info())
# train_csv = train_csv.drop(labels='TRAIN_28730',axis=0)
train_csv.iloc[28730, 3] = 'OWN'
test_csv.iloc[34486,7] = '기타'
# print(X.info())

# X.dropna(inplace=True)
# print(X.isnull())

# print(X.isnull().sum())
# print(test_csv.iloc[34486,7])



# X = lae.fit_transform(train_csv['대출등급'])
# X = lae.fit_transform(train_csv['주택소유상태'])
# X = lae.fit_transform(train_csv['대출목적'])


df = pd.DataFrame(train_csv)
df1 = pd.DataFrame(test_csv)

lae = LabelEncoder()

lae.fit(df['주택소유상태'])
df['주택소유상태'] = lae.transform(df['주택소유상태'])
df1['주택소유상태'] = lae.transform(df1['주택소유상태'])

# print(pd.value_counts(df['주택소유상태']))

lae.fit(df['대출목적'])
df['대출목적'] = lae.transform(df['대출목적'])
df1['대출목적'] = lae.transform(df1['대출목적'])

# print(pd.value_counts(df['대출목적']))
# print(pd.value_counts(df1['대출목적']))

lae.fit(df['대출기간'])
# df['대출기간'] = lae.transform(df['대출기간'])
# df1['대출기간'] = lae.transform(df1['대출기간'])
df['대출기간'] = df['대출기간'].replace({' 36 months' : 36 , ' 60 months' : 60 }).astype(int)         #  36 months -> 36, 60 months -> 60
df1['대출기간'] = df1['대출기간'].replace({' 36 months' : 36 , ' 60 months' : 60 }).astype(int)
# print(pd.value_counts(df['대출기간']))

lae.fit(df['근로기간'])
df['근로기간'] = lae.transform(df['근로기간'])
df1['근로기간'] = lae.transform(df1['근로기간'])

lae.fit(df['대출등급'])

# print(pd.value_counts(df['근로기간']))



# lae.fit(df['주택소유상태'])
# df1['주택소유상태'] = lae.transform(df1['주택소유상태'])
# df['주택소유상태'] = lae.transform(df['주택소유상태'])

# # print(pd.value_counts(df1['주택소유상태']))
# lae.fit(df['대출목적'])
# df1['대출목적'] = lae.transform(df1['대출목적'])
# # print(pd.value_counts(df1['대출목적']))

# lae.fit(df['대출기간'])
# df1['대출기간'] = lae.transform(df1['대출기간'])
# # print(pd.value_counts(df1['대출기간']))

# lae.fit(df['근로기간'])
# df1['근로기간'] = lae.transform(df1['근로기간'])
# print(pd.value_counts(df1['근로기간']))


# df1['대출등급'] = lae.transform(df1['대출등급'])

# print(train_csv)

# df = df[df.근로기간 !='unknown']


X = df.drop(['대출등급'], axis=1)
y = df['대출등급']

# mms = MinMaxScaler()
# mms.fit(X)
# X = mms.transform(X)
# df1 = mms.transform(df1)




# print(y.shape)              #(96294, )
# print(X)
# print(X['대출기간'].shape)

# print(df)       # (96293,14)
# print(df1)      # (64197, 13)
# print(y.shape)          #(96294, 7)


# test_csv['대출목적'] = lae.transform(test_csv['대출목적']) 
# print(df)
# print(df1)

# print(df['근로기간'])     #(96294, )                # 'unnkown' 삭제

y = y.values.reshape(-1, 1)
ohe = OneHotEncoder(sparse=False)
ohe.fit(y)
y1 = ohe.transform(y)
# y1 = OneHotEncoder(sparse=False).fit_transform(y) 
# # # print(y1)
# # # print(y1.shape)                 #
# # # print(X.shape)              #(96294, 13)

# # # print(train_csv.columns)
# # # Index(['대출금액', '대출기간', '근로기간', '주택소유상태', '연간소득', '부채_대비_소득_비율', '총계좌수', '대출목적',
# # #        '최근_2년간_연체_횟수', '총상환원금', '총상환이자', '총연체금액', '연체계좌수', '대출등급'], dtype='object')
# # # print(train_csv.info())
# # #  0   대출금액          96294 non-null  int64
# # #  1   대출기간          96294 non-null  object
# # #  2   근로기간          96294 non-null  object
# # #  3   주택소유상태        96294 non-null  object
# # #  4   연간소득          96294 non-null  int64
# # #  5   부채_대비_소득_비율   96294 non-null  float64
# # #  6   총계좌수          96294 non-null  int64
# # #  7   대출목적          96294 non-null  object
# # #  8   최근_2년간_연체_횟수  96294 non-null  int64
# # #  9   총상환원금         96294 non-null  int64
# # #  10  총상환이자         96294 non-null  float64
# # #  11  총연체금액         96294 non-null  float64
# # #  12  연체계좌수         96294 non-null  float64
# # #  13  대출등급          96294 non-null  object

# # # print(df)
# # # print(y)



X_train, X_test, y_train, y_test = train_test_split(X, y1, test_size=0.35, shuffle=True, random_state=3, stratify=y1)
mms = RobustScaler()
mms.fit(X_train)
X_train = mms.transform(X_train)
X_test = mms.transform(X_test)
df1 = mms.transform(df1)



# mms = MinMaxScaler()
# mms.fit(X_train)
# X_train = mms.transform(X_train)
# X_test = mms.transform(X_test)

# print(X_train)
# print(X_test)
# print(y_test)


# model = Sequential()
# model.add(Dense(19, input_shape= (13, ),activation='relu'))
# model.add(Dense(97,activation='relu'))
# model.add(Dense(9,activation='relu'))
# model.add(Dense(21,activation='relu'))
# model.add(Dense(16,activation='relu'))
# model.add(Dense(21,activation='relu'))
# model.add(Dense(7, activation='softmax'))

# mcp = ModelCheckpoint(monitor='val_loss', mode='min', verbose=1, save_best_only=True, filepath="c:\\_data\\_save\\MCP\\keras26_MCP11_dacon_loan.hdf5")    
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')
# es = EarlyStopping(monitor='acc', mode='max', patience=1000, verbose=20, restore_best_weights=True)
# model.fit(X_train, y_train, epochs=10000, batch_size=500, validation_split=0.15, callbacks=[es,mcp], verbose=2)

model= load_model("..\\_data\\_save\\MCP\\k26_11_dacon_loan_0117_1446_08332-0.9030-0.2746.hdf5")




results = model.evaluate(X_test, y_test)
print("ACC : ", results[1])

y_predict = model.predict(X_test) 
y_test = np.argmax(y_test, axis=1)
y_predict = np.argmax(y_predict, axis=1)
# y_test = ohe.inverse_transform(y_test)
# y_predict = ohe.inverse_transform(y_predict)
y_test = lae.inverse_transform(y_test)
y_predict = lae.inverse_transform(y_predict)

y_submit = model.predict(df1)  
y_submit = np.argmax(y_submit, axis=1)
# y_submit = ohe.inverse_transform(y_submit)
y_submit = lae.inverse_transform(y_submit)

# y_submit = pd.DataFrame(y_submit)
submission_csv['대출등급'] = y_submit
print(y_submit)

fs = f1_score(y_test, y_predict, average='macro')
print("f1_score : ", fs)

submission_csv.to_csv(path + "submission_0117_11_.csv", index=False)