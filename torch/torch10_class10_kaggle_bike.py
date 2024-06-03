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

# torch 데이터 만들기 1. x와 y를 합침
from torch.utils.data import TensorDataset, DataLoader
train_set = TensorDataset(x_train, y_train)
test_set = TensorDataset(x_test, y_test)

# torch 데이터 만들기 2. 배치 넣어줌.
print(len(train_set))#398
train_loader = DataLoader(train_set, batch_size=200, shuffle=True)
test_loader = DataLoader(test_set, batch_size=200, shuffle=False)

#2. 모델구성
class Model(nn.Module):
    def __init__(self, input_dim, output_dim):
        #super().__init__()
        super(Model, self).__init__()
        self.linear1 = nn.Linear(input_dim, 64)
        self.linear2 = nn.Linear(64, 32)
        self.linear3 = nn.Linear(32, 16)
        self.linear4 = nn.Linear(16, 8)
        self.linear5 = nn.Linear(8, output_dim)
        self.sigmoid = nn.Softmax()
        self.relu = nn.ReLU()
        return
    
    #순전파
    def forward(self, input_size):
        x = self.linear1(input_size)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        x = self.linear4(x)
        x = self.linear5(x)
        return x

model = Model(8, 1).to(DEVICE)


def RMSELoss(yhat,y):
    return torch.sqrt(torch.mean((yhat-y)**2))

#3. 컴파일, 훈련
criterion = RMSELoss
optimizer = optim.Adam(model.parameters(), lr = 0.001)
def train(model, criterion, optimizer, loader):   
    model.train()
    total_loss = 0
    for x_batch, y_batch in loader: 
        optimizer.zero_grad()
        hypothesis = model(x_batch) #예상치 값 (순전파)
        loss = criterion(hypothesis, y_batch) #예상값과 실제값 loss
        
        #역전파
        loss.backward() #기울기(gradient) 계산 (loss를 weight로 미분한 값)
        optimizer.step() # 가중치(w) 수정(weight 갱신)
        total_loss += loss.item()
    return total_loss / len(loader) #item 하면 numpy 데이터로 나옴

epochs = 200
for epoch in range(1, epochs + 1):
    loss = train(model, criterion, optimizer, train_loader)
    print('epoch {}, loss: {}'.format(epoch, loss)) #verbose

print("===================================")

#4. 평가, 예측
def evaluate(model, criterion, loader):
    model.eval() #평가모드
    total_loss = 0
    with torch.no_grad():
        for x_batch, y_batch in loader:
            hypothesis = model(x_batch)
            loss = criterion(hypothesis, y_batch)
            total_loss += loss.item()
    return total_loss / len(loader)

loss2 = evaluate(model, criterion, test_loader)
print("최종 loss : ", loss2)

y_pred = []
y_true = []

with torch.no_grad():
    for x_batch, y_batch in test_loader:
        result = model(x_batch)
        y_pred.append(result.cpu().numpy().squeeze())
        y_true.append(y_batch.cpu().numpy().squeeze())

y_pred = np.concatenate(y_pred, axis=0)
y_true = np.concatenate(y_true, axis=0)
print('r2 : ', r2_score(y_true, y_pred))