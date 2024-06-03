import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print('torch : ', torch.__version__, '사용 DEVICE :', DEVICE)

#1. 데이터 
path = 'C:/_data/kaggle/bike/'
train_csv =pd.read_csv(path + 'train.csv', index_col=0)
test_csv =pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sampleSubmission.csv')

x = train_csv.drop('count', axis=1).drop('casual', axis=1).drop('registered', axis=1)
y = train_csv['count']

x_train, x_test, y_train, y_test = train_test_split(x,y, shuffle=True, train_size=0.7, random_state=12345)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=42, shuffle=True)

x_train = torch.FloatTensor(np.array(x_train)).unsqueeze(1).to(DEVICE)
y_train = torch.FloatTensor(np.array(y_train)).unsqueeze(1).to(DEVICE)
x_test = torch.FloatTensor(np.array(x_test)).unsqueeze(1).to(DEVICE)
y_test = torch.FloatTensor(np.array(y_test)).unsqueeze(1).to(DEVICE)

print(x_train, y_train) #tensor([1., 2., 3.]) tensor([1., 2., 3.])

#2. 모델구성
# model = nn.Sequential(
#     nn.Linear(8, 64),
#     nn.Linear(64, 32),
#     nn.ReLU(),
#     nn.Linear(32, 16),
#     nn.Linear(16, 8),
#     nn.Linear(8, 1)
# ).to(DEVICE)

class Model(nn.Module):
    def __init__(self, input_dim, output_dim):
        #super().__init__()
        super(Model, self).__init__()
        self.linear1 = nn.Linear(input_dim, 64)
        self.linear2 = nn.Linear(64, 32)
        self.linear3 = nn.Linear(32, 16)
        self.linear4 = nn.Linear(16, output_dim)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        return
    
    #순전파
    def forward(self, input_size):
        x = self.linear1(input_size)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        x = self.linear4(x)
        return x
    
model = Model(8, 1).to(DEVICE)

def RMSELoss(yhat,y):
    return torch.sqrt(torch.mean((yhat-y)**2))

#3. 컴파일, 훈련
criterion = RMSELoss
optimizer = optim.Adam(model.parameters(), lr = 0.001)
def train(model, criterion, optimizer, x, y):    
    optimizer.zero_grad()
    hypothesis = model(x) 
    loss = criterion(hypothesis, y) 
    
    #역전파
    loss.backward() 
    optimizer.step() 
    return loss.item() 

epochs = 1000
for epoch in range(1, epochs + 1):
    loss = train(model, criterion, optimizer, x_train, y_train)
    print('epoch {}, loss: {}'.format(epoch, loss)) 
print("===================================")

#4. 평가, 예측
def evaluate(model, criterion, x, y):
    model.eval() #평가모드

    with torch.no_grad():
        y_predict = model(x)
        loss2 = criterion(y, y_predict)
    return loss2.item()

loss2 = evaluate(model, criterion, x_test, y_test)
print("최종 loss : ", loss2)
result = model(x_test).cpu().detach().numpy().squeeze()
r2 = r2_score(result, y_test.cpu().numpy().squeeze())
print('r2 : ', r2)