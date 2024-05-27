# 02 digits
# 03 fetch_covtype
# 04 dacon wine
# 05 dacon loan
# 06 kaggle 비만
# 07 load_diabetes
# 08 california
# 09 dacon ddarung
# 10 kaggle bike

# rmse, r2

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print('torch : ', torch.__version__, '사용 DEVICE :', DEVICE)

#1. 데이터 
path = "..\_data\dacon\ddarung\\"
train_csv = pd.read_csv(path + "train.csv", index_col='id')
# print(train_csv.shape)  # (1459, 10)
train_csv = train_csv.dropna()

test_csv = pd.read_csv(path + "test.csv", index_col='id')
# print(test_csv.shape)   # (715, 9)
test_csv.fillna(value=0, inplace=True)

submission_csv = pd.read_csv( path + "submission.csv")

x = train_csv.drop(['count'], axis=1)
y = np.array(train_csv['count']).reshape(-1, 1)
print(x.shape)
print(y.shape)
print(np.unique(y, return_counts=True))
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=42)


scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


x_train = torch.FloatTensor(x_train).to(DEVICE)
y_train = torch.FloatTensor(y_train).to(DEVICE)
x_test = torch.FloatTensor(x_test).to(DEVICE)
y_test = torch.FloatTensor(y_test).to(DEVICE)
print(y_train.shape)

#2. 모델구성
# model = Sequential()
# model.add(Dense(1, input_dim=1))
# model = nn.Linear(1, 1).to(DEVICE) #output, input
model = nn.Sequential(
    nn.Linear(x_train.shape[1], 64),
    nn.ReLU(),
    nn.Linear(64, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 1)
).to(DEVICE)

#3. 컴파일, 훈련
# model.compile(loss = 'mse', optimizer = 'adam')
criterion = nn.MSELoss()                #criterion : 표준
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
    y_pred = model(x_test).round()
    rmse = np.sqrt(mean_squared_error(y_test.cpu(), y_pred.cpu()))
    print('rmse : ', round(rmse, 5))


'''
최종 loss :  2386.071533203125
rmse :  48.84178
'''