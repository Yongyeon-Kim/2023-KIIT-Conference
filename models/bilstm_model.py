import pandas as pd
import numpy as np
from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import RMSprop


# trainset.csv에서 훈련 데이터 로드
train_data = pd.read_csv('../datasets/train/약관/trainset.csv')

# 데이터를 입력 특성(X_train)과 목표 레이블(y_train)로 분리
X_train = train_data.iloc[:, 0].values  # 첫 번째 열이 텍스트로 된 입력 특성이라고 가정
y_train = train_data.iloc[:, 1].values  # 두 번째 열이 목표 레이블이라고 가정

# validset.csv에서 검증 데이터 로드
valid_data = pd.read_csv('../datasets/valid/약관/vaildset.csv')

# 데이터를 입력 특성(X_valid)과 목표 레이블(y_valid)로 분리
X_valid = valid_data.iloc[:, 0].values  # 첫 번째 열이 텍스트로 된 입력 특성이라고 가정
y_valid = valid_data.iloc[:, 1].values  # 두 번째 열이 목표 레이블이라고 가정

# Define callbacks
early_stopping = EarlyStopping(patience=3, monitor='val_loss', restore_best_weights=True)
checkpoint = ModelCheckpoint('../models/log/BILSTM_best_model.h5', save_best_only=True, monitor='val_accuracy', mode='max')

# 텍스트 데이터 토큰화
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)

# 텍스트 데이터를 시퀀스로 변환
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_valid_seq = tokenizer.texts_to_sequences(X_valid)

# 시퀀스를 동일한 길이로 패딩
max_sequence_length = max(len(seq) for seq in X_train_seq + X_valid_seq)
X_train_padded = pad_sequences(X_train_seq, maxlen=max_sequence_length)
X_valid_padded = pad_sequences(X_valid_seq, maxlen=max_sequence_length)

# Bi-LSTM 모델 구성
model = Sequential()
model.add(Embedding(len(tokenizer.word_index) + 1, 200, input_length=max_sequence_length))
model.add(Bidirectional(LSTM(256, return_sequences=True)))
model.add(Bidirectional(LSTM(256)))
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 모델 컴파일
optimizer=RMSprop(learning_rate=0.0005)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
model.summary()

# Bi-LSTM 모델 훈련
# GRU 모델을 학습합니다
history = model.fit(X_train_padded, y_train, epochs=50, batch_size=256, validation_data=(X_valid_padded, y_valid),
                    callbacks=[early_stopping, checkpoint])

best_model = load_model('../models/log/BILSTM_best_model.h5')

# 검증 데이터에 대한 예측 수행
y_pred = (best_model.predict(X_valid_padded) > 0.5).astype("int32")

# Bi-LSTM 모델의 성능 평가
accuracy = accuracy_score(y_valid, y_pred)
print("Best 'Bi-LSTM' Model Accuracy:", accuracy)

with open("../models/log/Bi-LSTM.txt", "w") as file:
    # Write best model summary
    file.write("Best Model Summary:\n")
    best_model.summary(print_fn=lambda x: file.write(x + '\n'))

    # Write model performance
    file.write("\nModel Performance:\n")
    file.write(f"Accuracy: {accuracy}\n")