import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier

X, y = load_digits(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42, stratify=y)

ms = MinMaxScaler()
X_train = ms.fit_transform(X_train)
X_test = ms.transform(X_test)

xgb = XGBClassifier()
rf = RandomForestClassifier()
lr = LogisticRegression()

model = VotingClassifier(
    estimators=[('LR', lr), ('RF', rf), ('XGB', xgb)],
    # voting='hard',  # 디폴트
    voting='soft',
    
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("acc : ", model.score(X_test, y_test))
f1 = f1_score(y_test, y_pred, average='macro')
print("f1 : ", f1)

# acc :  0.9777777777777777
# f1 :  0.9774371947505609