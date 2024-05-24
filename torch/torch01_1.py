import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

#1. 데이터 
x = np.array([1,2,3])
y = np.array([1,2,3])

x = torch.FloatTensor(x).unsqueeze(1)
y = torch.FloatTensor(y).unsqueeze(1)

print(x, y) #tensor([1., 2., 3.]) tensor([1., 2., 3.])

#2. 모델구성
# model = Sequential()
# model.add(Dense(1, input_dim=1))
model = nn.Linear(1, 1) #output, input

#3. 컴파일, 훈련
# model.compile(loss = 'mse', optimizer = 'adam')
criterion = nn.MSELoss()                #criterion : 표준
# optimizer = optim.Adam(model.parameters(), lr = 0.01)
optimizer = optim.SGD(model.parameters(), lr = 0.1)

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
    
epochs = 200
for epoch in range(1, epochs + 1):
    loss = train(model, criterion, optimizer, x, y)
    print('epoch : {}, loss : {}'.format(epoch, loss))
    
print("######################################################")

#4. 평가, 예측
def evaluate(model, criterion, x, y):
    model.eval()    # 평가모드
    with torch.no_grad():
        loss = criterion(model(x), y)
    return loss.item()
print('최종 loss : ', evaluate(model, criterion, x, y))
result = model(torch.tensor([[4.]]))
print('최종 결과 : ', result.item())

