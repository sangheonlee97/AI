import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
a = pd.DataFrame({'class_id': ['a', 'a', 'a', 'a', 'a', 'a', 'b', 'b', 'b'],
                  'feature1': [1,2,3,4,5,6,7,8,9],})
print(a)


train_df, val_df = train_test_split(a, test_size=0.7, random_state=42, stratify=a['class_id'])
print(val_df['class_id'].value_counts())