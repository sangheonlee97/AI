import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Embedding, MultiHeadAttention, Dropout, LayerNormalization
from tensorflow.keras.models import Model

def positional_encoding(max_len, d_model):
    pos = tf.range(max_len, dtype=tf.float32)[:, tf.newaxis]
    i = tf.range(d_model, dtype=tf.float32)[tf.newaxis, :]
    angle_rads = pos / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
    sines = tf.math.sin(angle_rads[:, 0::2])  # 짝수 인덱스에 대한 sin
    cosines = tf.math.cos(angle_rads[:, 1::2])  # 홀수 인덱스에 대한 cos
    pos_encoding = tf.concat([sines, cosines], axis=-1)
    pos_encoding = pos_encoding[tf.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)

import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

# 데이터 로드
vocab_size = 10000
max_len = 200
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)

# 시퀀스 패딩
x_train = pad_sequences(x_train, maxlen=max_len, padding='post')
x_test = pad_sequences(x_test, maxlen=max_len, padding='post')

# 데이터 분할
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42)

# 모델 정의
def transformer_model(input_vocab_size, max_seq_len, d_model, num_heads, dff, num_layers):
    inputs = tf.keras.Input(shape=(max_seq_len,))
    
    # 임베딩 레이어
    embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)(inputs)
    
    # 포지셔널 인코딩
    pos_encoding = positional_encoding(max_seq_len, d_model)
    embedding += pos_encoding
    
    # 트랜스포머 레이어
    transformer_block = tf.keras.Sequential([
        tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.LayerNormalization(epsilon=1e-6),
        tf.keras.layers.Dense(dff, activation='relu'),
        tf.keras.layers.Dense(d_model),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.LayerNormalization(epsilon=1e-6)
    ])
    
    output = transformer_block(embedding)
    output = tf.keras.layers.GlobalAveragePooling1D()(output)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(output)
    
    model = tf.keras.Model(inputs=inputs, outputs=output)
    return model

# 하이퍼파라미터 설정
d_model = 128  # 임베딩 차원
num_heads = 4  # 어텐션 헤드의 수
dff = 512  # 피드포워드 네트워크의 은닉층 크기
num_layers = 4  # 인코더 및 디코더의 수

# 모델 생성
model = transformer_model(vocab_size, max_len, d_model, num_heads, dff, num_layers)

# 모델 컴파일
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 모델 훈련
model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=5, batch_size=64)

# 모델 평가
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {test_acc}")
