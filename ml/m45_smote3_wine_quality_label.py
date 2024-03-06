from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd

path = "../_data/dacon/wine/"

#1.데이터 가져오기
train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission_csv = pd.read_csv(path + "sample_submission.csv")

train_csv['type'] = train_csv['type'].replace({"white":1, "red":0})
test_csv['type'] = test_csv['type'].replace({"white":1, "red":0})

X = train_csv.drop(columns='quality')
y = train_csv['quality']

train = train_csv.copy()


g = train_csv.groupby('quality').size()

g_i = g.index
g_v = g.values
print("g_i : ", g_i)
print("g_v : ", g_v)

for i,v  in enumerate(train_csv['quality']):
    if v in g_i[np.where(g_v < 800)]:
        train.loc[i, 'quality'] = 0
    else:
        train.loc[i, 'quality'] = 1
        

g = train.groupby('quality').size()


yy = train['quality']


# X, _, y, _ = train_test_split(X,y, test_size=0.8, random_state=42, stratify=y)
print(np.unique(yy, return_counts=True)) # binary ce

from imblearn.over_sampling import SMOTE
smt = SMOTE(random_state=42, k_neighbors=4)

X, y = smt.fit_resample(X, yy)
print(np.unique(y, return_counts=True)) # binary ce

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = RandomForestClassifier(random_state=42)

model.fit(X_train, y_train, )

y_pred = model.predict(X_test)
f1 = f1_score(y_test, y_pred, average='macro')
print("f1 : ", f1)
print("acc : ", model.score(X_test, y_test))

# f1 :  0.6644580572571572
# acc :  0.9445454545454546

# f1 :  0.9614983321121552
# acc :  0.9615009746588694
