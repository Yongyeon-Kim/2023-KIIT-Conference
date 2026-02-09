import pandas as pd
from transformers import BertTokenizer, TFBertForSequenceClassification
from sklearn.metrics import accuracy_score
import tensorflow as tf
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

# 사전 훈련된 BERT 토크나이저 로드
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# 텍스트 데이터를 토큰화하고 인코딩

X_train_encoded = tokenizer.batch_encode_plus(X_train, padding=True, truncation=True, max_length=512, return_tensors='tf')
X_valid_encoded = tokenizer.batch_encode_plus(X_valid, padding=True, truncation=True, max_length=512, return_tensors='tf')

# 입력 특성을 TensorFlow 텐서로 변환
X_train_input_ids = X_train_encoded['input_ids']
X_train_attention_mask = X_train_encoded['attention_mask']
X_valid_input_ids = X_valid_encoded['input_ids']
X_valid_attention_mask = X_valid_encoded['attention_mask']

# 시퀀스 분류를 위한 BERT 모델 생성
model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# 모델 컴파일
optimizer = tf.keras.optimizers.Adam(1e-5)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
model.compile(optimizer=optimizer, loss=loss, metrics=[metric])
model.summary()

# BERT 모델 훈련
history = model.fit(
    {'input_ids': X_train_input_ids, 'attention_mask': X_train_attention_mask},
    y_train,
    epochs=30,
    batch_size=256,
    validation_data=({'input_ids': X_valid_input_ids, 'attention_mask': X_valid_attention_mask}, y_valid)
)
best_model = load_model('../models/log/BERT_best_model.h5')

# 검증 데이터에 대한 예측 수행
y_pred = best_model.predict({'input_ids': X_valid_input_ids, 'attention_mask': X_valid_attention_mask}).logits.argmax(axis=1)

# Bi-LSTM 모델의 성능 평가
accuracy = accuracy_score(y_valid, y_pred)
print("Best Model Accuracy:", accuracy)

with open("../models/log/BERT.txt", "w") as file:
    # Write best model summary
    file.write("Best Model Summary:\n")
    best_model.summary(print_fn=lambda x: file.write(x + '\n'))

    # Write model performance
    file.write("\nModel Performance:\n")
    file.write(f"Accuracy: {accuracy}\n")