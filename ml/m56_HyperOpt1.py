import numpy as np
import hyperopt
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
print(hyperopt.__version__)

search_space = {'X1' : hp.quniform('X1', -10, 10, 1),
                'X2' : hp.quniform('X2', -15, 15, 1)}

'''
hp.quniform(label, low, high, q) : label로 지정된 입력 값 변수 검색 공간을 최소값 low에서 최대값 high까지 q의 간격을 가지고 설정
hp.uniform(label, low, high) : 최소값 low에서 최대값 high까지 정규분포 형태의 검색 공간 설정
hp.randint(label, upper) : 0부터 최대값 upper까지 random한 정수값으로 검색 공간 설정
hp.loguniform(label, low, high)  : exp(uniform(low, high))값을 반환하며, 반환값의 log변환된 값은 정규분포 형태를 가지는 검색 공간 설정
'''

def objective_func(search_space):
    X1 = search_space['X1']
    X2 = search_space['X2']
    retval = X1**2 - 20*X2
    
    return retval

trial_val = Trials()

best = fmin(
    fn=objective_func,
    space=search_space,
    algo=tpe.suggest,   # default
    max_evals=20,       # 서치 횟수
    trials=trial_val,
    rstate=np.random.default_rng(seed=10)
)

print(best)
import pandas as pd
arr = []
loss = []
x1 = []
x2 = []
for i in range(20):
    loss.append(trial_val.results[i]['loss'])
    x1.append(trial_val.vals['X1'][i])
    x2.append(trial_val.vals['X2'][i])
arr = np.array([loss, x1, x2])
arr = np.array(arr).T.reshape(-1,3)
print(pd.DataFrame(arr, columns=['loss', 'x1', 'x2']))