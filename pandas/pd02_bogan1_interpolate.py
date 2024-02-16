'''
결측치 처리
1. 행 또는 열 삭제
2. 임의의 값을 넣어준다
    fillna, ffill, bfill, 중위값, 평균값...
3. 보간 :  interploate
4. 모델 predict
5. 부스팅 계열 : 통상 결측치 이상치에 대해 자유롭다


'''



import numpy as np
import pandas as pd
from datetime import datetime

dates = ['2/16/2024', '2/17/2024', '2/18/2024', 
         '2/19/2024', '2/20/2024', '2/21/2024']
dates = pd.to_datetime(dates)
print(dates)
print("============================")
ts = pd.Series([2, np.nan, np.nan,
                8, 10, np.nan], index=dates)
print(ts)
print("============================")
ts = ts.interpolate()
print(ts)