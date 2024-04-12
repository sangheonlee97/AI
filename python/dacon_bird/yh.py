from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import numpy as np
from function_package import merge_image
import tensorflow as tf
import random

# RANDOM_STATE = 47
RANDOM_STATE = np.random.randint(1,10000)
BATCH_SIZE = int(1000)
IMAGE_SIZE = int(64)   

random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

path = "../_data/dacon/bird/"

# xy_data_gen = ImageDataGenerator(
#     rescale=1./255,
# )

# xy_data = xy_data_gen.flow_from_directory(
#     path,
#     target_size=(IMAGE_SIZE,IMAGE_SIZE),
#     batch_size=BATCH_SIZE,
#     class_mode='sparse',
#     shuffle=False
# )

# x, y = merge_image(xy_data)
# print(x.shape, y.shape)

# data_dict = {0:[],1:[],2:[]}
# for data, data_class in zip(x,y):
#     data_dict[data_class].append(data)

# test = data_dict[0]
# x_data = data_dict[1]
# upscale_x_data = data_dict[2]

import pickle
# with open(path+'test.pkl','wb') as test_f:
#     pickle.dump(test,test_f)
# with open(path+'x_data.pkl','wb') as x_data_f:
#     pickle.dump(x_data,x_data_f)
# with open(path+'upscale_x_data.pkl','wb') as upscale_x_data_f:
#     pickle.dump(upscale_x_data,upscale_x_data_f)

with open(path+'test.pkl','rb') as test_f:
    test = pickle.load(test_f)
with open(path+'x_data.pkl','rb') as x_data_f:
    x_data = pickle.load(x_data_f)
with open(path+'upscale_x_data.pkl','rb') as upscale_x_data_f:
    upscale_x_data = pickle.load(upscale_x_data_f)

print(len(x_data),len(test),len(upscale_x_data))
# 15834 6786 15834
print(f"data_{IMAGE_SIZE}px_b file created")

train_csv = pd.read_csv(path+'train.csv')

y_data = train_csv['label'].to_numpy()

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split

label_encoder = LabelEncoder()
y_data = label_encoder.fit_transform(y_data.reshape(-1,1))
print(np.unique(y_data,return_counts=True))
# (array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
#        17, 18, 19, 20, 21, 22, 23, 24]), array([613, 642, 641, 657, 630, 653, 643, 550, 629, 661, 631, 630, 645,
#        637, 431, 639, 658, 663, 663, 671, 616, 657, 659, 666, 649],
y_data = OneHotEncoder(sparse=False).fit_transform(y_data.reshape(-1,1))

x_train, x_test, y_train, y_test = train_test_split(np.array(x_data),np.array(y_data),train_size=0.8,random_state=RANDOM_STATE,stratify=y_data)
print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)
# (12667, 64, 64, 3) (3167, 64, 64, 3) (12667, 25) (3167, 25)

from keras.applications import EfficientNetV2M, VGG16, VGG19, ResNet101V2
from keras.models import Model
from keras.layers import Input, Dense, Flatten, Dropout

# base_model = EfficientNetV2M(include_top=False,input_shape=(64,64,3),classes=25,weights='imagenet')
# base_model = VGG16(weights='imagenet', include_top=False, input_shape=(64,64,3))
base_model = ResNet101V2(weights='imagenet', include_top=False, input_shape=(64,64,3))
base_model.trainable = False
inputs = base_model.input

fl = Flatten()(base_model.output)
d1 = Dense(512,activation='swish',name='d1')(fl)
dr1 = Dropout(0.05)(d1)
outputs = Dense(25,activation='softmax',name='final')(dr1)
model = Model(inputs=[inputs],outputs=[outputs])

from keras.optimizers import Adam
model.compile(optimizer=Adam(learning_rate=0.01),loss='categorical_crossentropy',metrics='acc')
model.summary()

from keras.callbacks import EarlyStopping, ReduceLROnPlateau
es = EarlyStopping(monitor='val_loss',mode='auto',patience=30,restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss',mode='auto',patience=50,factor=0.7)
model.fit(x_train,y_train,epochs=30,batch_size=128,validation_split=0.2,verbose=1,callbacks=[es,reduce_lr])

score = model.evaluate(x_test,y_test)
print(f"LOSS: {score[0]}\nACC:  {score[1]}")

test = np.array(test)
print("test.shape", test.shape)
pred = np.argmax(model.predict(test),axis=1)
print(pred.shape)

pred = label_encoder.inverse_transform(pred)

submit_csv = pd.read_csv(path+"sample_submission.csv")
submit_csv['label'] = pred
print(submit_csv.head)
submit_csv.to_csv(path+f'submit/ResNet101_acc{score[1]:.6f}_random{RANDOM_STATE}.csv',index=False)