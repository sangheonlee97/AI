import numpy as np
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout, Input
from sklearn.model_selection import RandomizedSearchCV
(X_train, y_train), (X_test, y_test) = mnist.load_data()
print(X_train.shape)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1] , X_train.shape[2], -1).astype('float32')/255.
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], -1).astype('float32')/255.
print(X_train.shape)
print(y_train.shape)
def build_model(drop=.5, input_shape=(28, 28, 1), opti='adam', lr=0.001, node1=64, node2=32, node3=16, activation='relu'):
    inp = Input(input_shape)
    l1 = Conv2D(node1, (3, 3),activation=activation)(inp)
    l1 = Dropout(drop)(l1)
    l2 = Conv2D(node2, (3, 3), activation=activation)(l1)
    l2 = Dropout(drop)(l2)
    l3 = Conv2D(node3, (3, 3), activation=activation)(l2)
    l3 = Dropout(drop)(l3)
    l3 = Flatten()(l3)
    l4 = Dense(32, activation='swish')(l3)
    outp = Dense(10, activation='softmax')(l4)
    model = Model(inputs=inp, outputs=outp)
    model.compile(optimizer=opti, loss='sparse_categorical_crossentropy', metrics=['acc'])
    
    return model

def create_hyperparameter():
    batchs = [100, 200, 300, 400, 500]
    optimizers = ['adam', 'rmsprop', 'adadelta']
    dropouts = [0.2, 0.3, 0.4, 0.5]
    activations = ['relu', 'elu', 'swish', 'linear']
    node1 = [64, 32, 16]
    node2 = [64, 32, 16]
    node3 = [64, 32, 16]
    return {'batch_size' : batchs,
            'opti' : optimizers,
            'drop' : dropouts,
            'activation' : activations,
            'node1' : node1,
            'node2' : node2,
            'node3' : node3,
            }
    
hyperparameters = create_hyperparameter()
from keras.wrappers.scikit_learn import KerasClassifier
from keras.callbacks import EarlyStopping, ModelCheckpoint

keras_model = KerasClassifier(build_fn=build_model, verbose=1)
model = RandomizedSearchCV(keras_model, hyperparameters, cv=3, n_iter=5, random_state=42, verbose=1)
es = EarlyStopping(monitor='loss', patience=2, mode='min', restore_best_weights=True)
mcp = ModelCheckpoint('./abcd.hdf5', monitor='loss', mode='min')
model.fit(X_train, y_train, epochs=10, callbacks=[es, mcp])

print('model.best_params_ : ', model.best_params_)
print('model.best_estimator_ : ', model.best_estimator_)
print('model.best_score_ : ', model.best_score_)
print('model.score : ', model.score(X_test, y_test))

# model.best_params_ :  {'opti': 'rmsprop', 'node3': 64, 'node2': 16, 'node1': 32, 'drop': 0.4, 'batch_size': 200, 'activation': 'swish'}
# model.best_estimator_ :  <keras.wrappers.scikit_learn.KerasClassifier object at 0x0000021736A625E0>
# model.best_score_ :  0.9803333282470703
# 50/50 [==============================] - 0s 2ms/step - loss: 0.0480 - acc: 0.9868
# model.score :  0.9868000149726868