import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Model, Sequential
from keras.layers import Dense, Input, concatenate, Concatenate, Reshape
X1_datasets = np.array([range(100), range(301, 401)]).T   # 삼전 종가, 하닉 종가
X2_datasets = np.array([range(101,201), range(411,511), range(150,250)]).T    # 원유, 환율, 금 시세
X3_datasets = np.array([range(100), range(301,401), range(77,177), range(33,133)]).T  # blah blah

print(X1_datasets.shape, X2_datasets.shape)

# 1
y = np.array(range(3001, 3101))     # bitcoin 종가
X1_train, X1_test, X2_train, X2_test, X3_train, X3_test, y_train, y_test = train_test_split(
    X1_datasets, X2_datasets, X3_datasets, y,
    test_size=0.3, random_state=123,
)

# 2 1
ip1 = Input(shape=(2,))
d1 = Dense(10, activation='relu', name='bit1')(ip1)
d2 = Dense(10, activation='relu', name='bit2')(d1)
d3 = Dense(10, activation='relu', name='bit3')(d2)
op1 = Dense(10, activation='relu', name='bit4')(d3)
# op1 = Reshape((2,5))(op1)

# model1 = Model(inputs=ip1, outputs=op1)
# model1.summary()

# 2 2
ip11 = Input(shape=(3,))
d11 = Dense(100, activation='relu', name='bit11')(ip11)
d21 = Dense(100, activation='relu', name='bit12')(d11)
d31 = Dense(100, activation='relu', name='bit13')(d21)
op11 = Dense(5, activation='relu', name='bit14')(d31)
# op11 = Reshape((2,4))(op11)

# model2 = Model(inputs=ip11, outputs=op11)
# model2.summary()

# 2 3
ip111 = Input(shape=(4,))
d111 = Dense(10, activation='relu')(ip111)
d112 = Dense(10, activation='relu')(d111)
d113 = Dense(10, activation='relu')(d112)
d114 = Dense(10, activation='relu')(d113)
op111 = Dense(6, activation='relu')(d114)


# 2concatenate
merge1 = concatenate([op1, op11, op111], name='mg1')
merge2 = Dense(7, name='mg2')(merge1)
merge3 = Dense(11, name='mg3')(merge2)
last_output = Dense(1, name='last')(merge3)

model = Model(inputs=[ip1, ip11, ip111], outputs=last_output)

model.summary()

# 3 
model.compile(loss='mse', optimizer='adam')
model.fit([X1_train, X2_train, X3_train], y_train, epochs=1000, batch_size=35)

res = model.evaluate([X1_test, X2_test, X3_test], y_test)
print("loss : ", res)
