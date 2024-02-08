import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, KFold

datasets = load_iris()
df = pd.DataFrame(datasets.data, columns=datasets.feature_names)
print(df)   # 150, 4
n_splits = 3

kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
# kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

for train_idx, val_idx in kfold.split(df):
    print("=========================")
    print(train_idx, "\n" ,val_idx)
    print(len(train_idx), len(val_idx))
    print("=========================")
