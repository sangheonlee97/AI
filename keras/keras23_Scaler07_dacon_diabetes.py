import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

path = "..\_data\dacon\diabetes\\"
######1. data
df_train = pd.read_csv(path + "train.csv", index_col=0)
df_test = pd.read_csv(path + "test.csv", index_col=0)
df_sub = pd.read_csv(path + "sample_submission.csv")

# print(df_train.shape)   # (652, 9)
# print(df_test.shape)   # (116, 8)
# print(df_train.info)

X = df_train.drop(['Outcome'], axis=1)
y = df_train['Outcome']

# print(X.shape)      # (652, 8)
# print(y.shape)      # (652, )


# print(X['Pregnancies'][X['Pregnancies']==0].count()) # 87
# print(X['Glucose'][X['Glucose']==0].count()) # 4
# print(X['BloodPressure'][X['BloodPressure']==0].count()) # 30
# print(X['SkinThickness'][X['SkinThickness']==0].count()) # 195
# print(X['Insulin'][X['Insulin']==0].count()) # 318 결측치가 많은건지, 실제 값이 0인건지 분간이 안됨;
# print(X['BMI'][X['BMI']==0].count()) # 7
# print(X['DiabetesPedigreeFunction'][X['DiabetesPedigreeFunction']==0].count()) # 0
# print(X['Age'][X['Age']==0].count()) # 0


# X = X.drop(['SkinThickness'], axis=1)
X = X.drop(['Insulin'], axis=1)
# df_test = df_test.drop(['SkinThickness'], axis=1)
df_test = df_test.drop(['Insulin'], axis=1)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.11, random_state=1062888800)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
######2. model
model = Sequential()
model.add(Dense(20, input_dim=7, activation='sigmoid'))
model.add(Dense(30, activation='sigmoid'))
model.add(Dense(20, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))

######3. compile, fit
model.compile(loss="binary_crossentropy", optimizer='adam', metrics=['accuracy'])
es = EarlyStopping(monitor='val_loss', patience=170, mode='min', verbose=1, restore_best_weights=True)
model.fit(X_train, y_train, epochs=1000, batch_size=15, validation_split=0.11, callbacks=[es])

######4. predict
y_pred = model.predict(X_test)
y_pred = y_pred.round()
y_sub = model.predict(df_test)
y_sub = y_sub.round()

df_sub['Outcome'] = y_sub
df_sub.to_csv(path + "submisson_0110_real.csv", index=False )

print(y_pred)
acc = accuracy_score(y_test, y_pred)
print("acc : ", acc)
# print(type(y_sub))
# print(y_sub.shape)

# acc :  0.875                  // 0.784
# ts :  0.11
# rs :  1062888800
# p :  149
# bs :  23
# acc :  0.875







# acc :  0.8472222222222222     // 0.818
# ts :  0.11
# rs :  1062888800
# p :  170
# bs :  18
# acc :  0.8472222222222222





