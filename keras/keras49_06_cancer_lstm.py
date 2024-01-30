from sklearn.datasets import load_breast_cancer
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, LSTM
from keras.callbacks import EarlyStopping
datasets = load_breast_cancer()
X = datasets.data
y = datasets.target

print(X.shape, y.shape) # (569, 30) (569,)
X = X.reshape(569, 10, 3)
y = y.reshape(569, 1)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, stratify=y, random_state=42)

model = Sequential()
model.add(LSTM(30, input_shape=(10,3), activation='relu'))
model.add(Flatten())
model.add(Dense(500, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
es = EarlyStopping(monitor='val_loss', mode='min', patience=50, restore_best_weights=True, verbose=1)
model.fit(X_train, y_train, epochs=1000, batch_size=32, validation_split=0.2, callbacks=[es])

result = model.evaluate(X_test, y_test)

print("acc : ", result[1])

# cnn이 구림 0.928
# LSTM : 0.947