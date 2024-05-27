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
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print('torch : ', torch.__version__, '사용 DEVICE :', DEVICE)

#1. 데이터 
x , y = dataset = fetch_covtype(return_X_y=True)
print(x.shape)
print(y.shape)
y = y - 1

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=42, stratify=y)


scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
print(np.unique(y_train, return_counts=True))
print(y_train.max())

x_train = torch.FloatTensor(x_train).to(DEVICE)
x_test = torch.FloatTensor(x_test).to(DEVICE)
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)
print(y_train.shape)

ohe = OneHotEncoder(sparse_output=False)
y_train = ohe.fit_transform(y_train)
y_test = ohe.transform(y_test)

y_train = torch.FloatTensor(y_train).to(DEVICE)
y_test = torch.FloatTensor(y_test).to(DEVICE)


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
    nn.Linear(32, 7),
    nn.Softmax()
).to(DEVICE)

#3. 컴파일, 훈련
# model.compile(loss = 'mse', optimizer = 'adam')
criterion = nn.CrossEntropyLoss()              #criterion : 표준
optimizer = optim.Adam(model.parameters(), lr = 0.001)
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
최종 loss :  1.3997105360031128
acc :  0.76613
'''