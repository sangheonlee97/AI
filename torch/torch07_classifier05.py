# 01 cancer
# 02 digits
# 03 fetch_covtype
# 04 dacon wine
# 05 dacon loan
# 06 kaggle 비만
# 07 load_diabetes
# 08 california
# 09 dacon ddarung
# 10 kaggle bike

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, RobustScaler
from sklearn.metrics import accuracy_score
import pandas as pd

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print('torch : ', torch.__version__, '사용 DEVICE :', DEVICE)

#1. 데이터 
path = "..\\_data\\dacon\\daechul\\"

train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission_csv = pd.read_csv(path + "sample_submission.csv")


train_csv.iloc[28730, 3] = 'OWN'
test_csv.iloc[34486,7] = '기타'


df = pd.DataFrame(train_csv)
df1 = pd.DataFrame(test_csv)

lae = LabelEncoder()

lae.fit(df['주택소유상태'])
df['주택소유상태'] = lae.transform(df['주택소유상태'])
df1['주택소유상태'] = lae.transform(df1['주택소유상태'])


lae.fit(df['대출목적'])
df['대출목적'] = lae.transform(df['대출목적'])
df1['대출목적'] = lae.transform(df1['대출목적'])



lae.fit(df['대출기간'])
df['대출기간'] = df['대출기간'].replace({' 36 months' : 36 , ' 60 months' : 60 }).astype(int)         #  36 months -> 36, 60 months -> 60
df1['대출기간'] = df1['대출기간'].replace({' 36 months' : 36 , ' 60 months' : 60 }).astype(int)


lae.fit(df['근로기간'])
df['근로기간'] = lae.transform(df['근로기간'])
df1['근로기간'] = lae.transform(df1['근로기간'])

lae.fit(df['대출등급'])


X = df.drop(['대출등급'], axis=1)
y = df['대출등급']

y = y.values.reshape(-1, 1)
ohe = OneHotEncoder(sparse_output=False)
ohe.fit(y)
y1 = ohe.transform(y)



x_train, x_test, y_train, y_test = train_test_split(X, y1, test_size=0.2, shuffle=True, random_state=42, stratify=y1)

rbs = RobustScaler()
rbs.fit(x_train)
x_train = rbs.transform(x_train)
x_test = rbs.transform(x_test)
df1 = rbs.transform(df1)

x_train = torch.FloatTensor(np.array(x_train)).to(DEVICE)
x_test = torch.FloatTensor(np.array(x_test)).to(DEVICE)


y_train = torch.FloatTensor(y_train).to(DEVICE)
y_test = torch.FloatTensor(y_test).to(DEVICE)


#2. 모델구성
# model = Sequential()
# model.add(Dense(1, input_dim=1))
# model = nn.Linear(1, 1).to(DEVICE) #output, input
model = nn.Sequential(
    nn.Linear(13, 64),
    nn.ReLU(),
    nn.Linear(64, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 7)
).to(DEVICE)

#3. 컴파일, 훈련
# model.compile(loss = 'mse', optimizer = 'adam')
criterion = nn.CrossEntropyLoss()              #criterion : 표준
optimizer = optim.Adam(model.parameters(), lr = 0.01)
# optimizer = optim.SGD(model.parameters(), lr = 0.001)

# model.fit(x,y, epochs = 100, batch_size=1)
def train(model, criterion, optimizer, x, y):
    # model.train()   #훈련모드 default

    optimizer.zero_grad()
    # w = w - lr * (loss를 weight로 미분한 값)
    hypothesis = model(x) #예상치 값 (순전파)
    loss = criterion(hypothesis, y) #예상값과 실제값 loss

    #역전파
    loss.backward() #기울기(gradient) 계산 (loss를 weight로 미분한 값)
    optimizer.step() # 가중치(w) 수정(weight 갱신)
    return loss.item() #item 하면 numpy 데이터로 나옴

epochs = 1000
for epoch in range(1, epochs + 1):
    loss = train(model, criterion, optimizer, x_train, y_train)
    print('epoch {}, loss: {}'.format(epoch, loss)) #verbose

print("===================================")

#4. 평가, 예측
# loss = model.evaluate(x,y)
def evaluate(model, criterion, x, y):
    model.eval() #평가모드

    with torch.no_grad():
        y_predict = model(x)
        loss2 = criterion(y_predict, y)
    return loss2.item()

loss2 = evaluate(model, criterion, x_test, y_test)
print("최종 loss : ", loss2)

with torch.no_grad():
    y_pred = model(x_test)
    y_pred = np.argmax(y_pred.cpu(), axis=1)
    y_test = np.argmax(y_test.cpu(), axis=1)
    acc = accuracy_score(y_test, y_pred)
    print('acc : ', round(acc, 5))


'''
최종 loss :  0.45870235562324524
acc :  0.84013
'''