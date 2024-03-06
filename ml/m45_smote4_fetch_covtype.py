import numpy as np
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

X, y = fetch_covtype(return_X_y=True)

print(X.shape)
print(np.unique(y, return_counts=True))

# X, _, y, _ = train_test_split(X, y, train_size=0.1, random_state=42, stratify=y)
print(np.unique(y, return_counts=True))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
from imblearn.over_sampling import SMOTE

smt = SMOTE()
# X_train, y_train = smt.fit_resample(X_train, y_train)
print(np.unique(y_train, return_counts=True))

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
f1 = f1_score(y_test, y_pred, average='macro')
print("acc : ", model.score(X_test, y_test))
print("f1 : ", f1)


# acc :  0.8817657688667069
# f1 :  0.8100147334457833

# acc :  0.8871869890715085
# f1 :  0.8428638676992627