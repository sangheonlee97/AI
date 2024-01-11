import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

path = "..\_data\dacon\diabetes\\"

df1 = pd.read_csv(path + "submission_0110_2024110184418.csv", index_col=0)
df2 = pd.read_csv(path + "submission_0110_2024110184511.csv", index_col=0)
df3 = pd.read_csv(path + "submission_0110_2024110184532.csv", index_col=0)
df4 = pd.read_csv(path + "submission_0110_2024110184556.csv", index_col=0)
df5 = pd.read_csv(path + "submission_0110_2024110184620.csv", index_col=0)
df6 = pd.read_csv(path + "submission_0110_2024110184639.csv", index_col=0)
df7 = pd.read_csv(path + "submission_0110_2024110184652.csv", index_col=0)
df8 = pd.read_csv(path + "submission_0110_2024110184721.csv", index_col=0)
df9 = pd.read_csv(path + "submission_0110_2024110184731.csv", index_col=0)
df10 = pd.read_csv(path + "submission_0110_2024110184746.csv", index_col=0)
df11 = pd.read_csv(path + "submission_0110_2024110184816.csv", index_col=0)
df12 = pd.read_csv(path + "submission_0110_2024110184836.csv", index_col=0)
df13 = pd.read_csv(path + "submission_0110_2024110184848.csv", index_col=0)
df14 = pd.read_csv(path + "submission_0110_202411018485.csv", index_col=0)
df15 = pd.read_csv(path + "submission_0110_2024110184919.csv", index_col=0)
df16 = pd.read_csv(path + "submission_0110_2024110184927.csv", index_col=0)
df17 = pd.read_csv(path + "submission_0110_2024110184937.csv", index_col=0)
df18 = pd.read_csv(path + "submisson_0110_11.csv", index_col=0)
df19 = pd.read_csv(path + "submisson_0110_22.csv", index_col=0)
df20 = pd.read_csv(path + "submission_0110_202411018494.csv", index_col=0)
df21 = pd.read_csv(path + "submission_0110_2024110184946.csv", index_col=0)
df22 = pd.read_csv(path + "submission_0110_2024110185123.csv", index_col=0)
df23 = pd.read_csv(path + "submission_0110_2024110185212.csv", index_col=0)
df24 = pd.read_csv(path + "submission_0110_2024110185235.csv", index_col=0)
df25 = pd.read_csv(path + "submisson_0110_3_3.csv", index_col=0)
df26 = pd.read_csv(path + "submisson_0110_3_6.csv", index_col=0)
df27 = pd.read_csv(path + "submisson_0110_real.csv", index_col=0)
df28 = pd.read_csv(path + "submission_0110_2024110185255.csv", index_col=0)
df29 = pd.read_csv(path + "submission_0110_2024110185311.csv", index_col=0)
# df = pd.read_csv(path + "", index_col=0)
# df = pd.read_csv(path + "", index_col=0)
# df = pd.read_csv(path + "", index_col=0)
# df = pd.read_csv(path + "", index_col=0)
# df = pd.read_csv(path + "", index_col=0)
# df = pd.read_csv(path + "", index_col=0)
# df = pd.read_csv(path + "", index_col=0)
# df = pd.read_csv(path + "", index_col=0)
# df = pd.read_csv(path + "", index_col=0)
# df = pd.read_csv(path + "", index_col=0)
# df = pd.read_csv(path + "", index_col=0)
# df = pd.read_csv(path + "", index_col=0)
# df = pd.read_csv(path + "", index_col=0)
# df = pd.read_csv(path + "", index_col=0)
# df = pd.read_csv(path + "", index_col=0)
# df = pd.read_csv(path + "", index_col=0)

df_sub = pd.read_csv(path + "sample_submission.csv")

df1 = df1.T
df2 = df2.T
df3 = df3.T
df4 = df4.T
df5 = df5.T
df6 = df6.T
df7 = df7.T
df8 = df8.T
df9 = df9.T
df10 = df10.T
df11 = df11.T
df12 = df12.T
df13 = df13.T
df14 = df14.T
df15 = df15.T
df16 = df16.T
df17 = df17.T
df18 = df18.T
df19 = df19.T
df20 = df20.T
df21 = df21.T
df22 = df22.T
df23 = df23.T
df24 = df24.T
df25 = df25.T
df26 = df26.T
df27 = df27.T
df28 = df28.T
df29 = df29.T

df30 = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8, df9, df10, df11, df12, df13, df14, df15, df16, df17, df18, df19, df20, df21, df22, df23, df24, df25, df26, df27, df28, df29])
# for i in range(116):
#     a = pd.value_counts(df20.iloc[:,i])
#     print("@@@@@@@@@@@@@@@")
    
#     print(a)
#     print("@@@@@@@@@@@@@@@")

idx = 0
# print(pd.value_counts(df20.iloc[:,0]).sort_index()) # 첫번째 칸의 0의 갯수
# dja = pd.value_counts(df20.iloc[:,0]).sort_index()
# print("@@@@@@@@@@@@@@@@@@@")
# print(dja.index[0]) # 0.0
# print(dja.index[0]) # 1.0 sort_index()안하면

# for i in range(116):
#     print(pd.value_counts(df20.iloc[:,idx]).sort_index())
#     idx += 1
# print(a.shape)

result = []
for i in range(116):
    f = pd.value_counts(df30.iloc[:,i]).sort_index()
    print(f)
    if f.iloc[0] == 29:
        result.append(f.index[0])
    elif f.iloc[0] > 14:
        result.append(f.index[0])
    else:
        result.append(f.index[1])
# print(df20)







result = pd.DataFrame(result)
print(result.shape) #(116, 1)
print(type(result)) #<class 'pandas.core.frame.DataFrame'>

df_sub['Outcome'] = result
df_sub.to_csv(path + "submisson_0111_nogada_2.csv", index=False )













#############################################
# df4 = pd.concat([df1, df2, df3])

# print(pd.value_counts(df4.iloc[:,0]))
#############################################

# <class 'numpy.ndarray'>
# (116, 1)