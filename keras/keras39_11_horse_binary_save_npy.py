from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pandas as pd

data_generator = ImageDataGenerator(
    rescale=1./255
    # horizontal_flip=True,   # 수평 뒤집기
    # vertical_flip=True,     # 수직 뒤집기
    # width_shift_range=0.1,  # 가로로 평행이동
    # height_shift_range=0.1, # 세로로 평행이동
    # rotation_range=5,       # 정해진 각도만큼 이미지를 회전
    # zoom_range=1.2,         # 축소 또는 확대
    # shear_range=0.7,        # 좌표 하나를 고정시키고 다른 몇 개의 좌표를 이동시키는 변환
    # fill_mode='nearest',    # 빈 공간을 근사치로 채움,    
)


path = "..\\_data\\image\\horse_human\\"
data = data_generator.flow_from_directory(
                                    path,
                                    target_size=(300,300),
                                    batch_size=100,
                                    # class_mode='binary', ### 다중분류할 땐 categorical을 써야하는데 이게 디폴트 값이라 주석처리해버린 모습
                                    shuffle=True
                                    )

X = []
y = []

for i in range(len(data)):
    batch = data.next()
    X.append(batch[0])
    y.append(batch[1])
X = np.concatenate(X, axis=0)
y = np.concatenate(y, axis=0)


np_path = "..\\_data\\_save_npy\\horse_human\\"

np.save(np_path + 'horse_human_X.npy', arr=X)
np.save(np_path + 'horse_human_y.npy', arr=y)