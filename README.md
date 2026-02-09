# 2023 KIIT Conference - 약관 데이터 분석 및 모델링

## 프로젝트 개요

이 프로젝트는 2023년 KIIT 컨퍼런스에서 발표된 약관 데이터를 분석하고, 이를 기반으로 딥러닝 모델을 구축하여 약관의 유리/불리 여부를 분류하는 것을 목표로 합니다.

## 디렉토리 구조

- `datasets/`: 학습 및 검증 데이터셋을 포함합니다.
  - `train.csv`: 훈련 데이터셋
  - `valid.csv`: 검증 데이터셋
- `models/`: 모델 정의 및 학습 관련 스크립트와 학습된 모델 파일을 포함합니다.
  - `log/`: 모델 학습 로그 및 저장된 최적 모델 파일을 포함합니다.
- `data_preprocessing.py`: JSON 형식의 원본 데이터를 CSV 형식으로 전처리하는 스크립트.
- `bert_model.py`: BERT 모델을 이용한 학습 및 평가 스크립트.
- `bilstm_model.py`: Bi-LSTM 모델을 이용한 학습 및 평가 스크립트.

## 시작하기

### 1. 환경 설정

필요한 라이브러리를 설치합니다.

```bash
pip install pandas numpy tensorflow scikit-learn transformers keras_preprocessing
```

### 2. 데이터 전처리

원본 JSON 데이터를 `train.csv` 및 `valid.csv` 파일로 변환합니다.

```bash
python data_preprocessing.py
```

### 3. 모델 학습 및 평가

각 모델 스크립트를 실행하여 모델을 학습하고 평가할 수 있습니다.

#### BERT 모델

```bash
python bert_model.py
```

#### Bi-LSTM 모델

```bash
python bilstm_model.py
```

## 라이센스

이 프로젝트는 MIT 라이센스를 따릅니다. 자세한 내용은 `LICENSE` 파일을 참조하십시오.
