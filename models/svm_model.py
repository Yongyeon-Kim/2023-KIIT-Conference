import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# trainset.csv에서 훈련 데이터 불러오기
train_data = pd.read_csv('datasets/train/약관/trainset.csv')

# 데이터를 입력 특성(X_train)과 타깃 레이블(y_train)로 분리하기
X_train = train_data.iloc[:, 0].values  # 첫 번째 열이 텍스트로 된 입력 특성인 경우
y_train = train_data.iloc[:, 1].values  # 두 번째 열이 타깃 레이블인 경우

# validset.csv에서 검증 데이터 불러오기
valid_data = pd.read_csv('datasets/valid/약관/vaildset.csv')

# 검증 데이터를 입력 특성(X_valid)과 타깃 레이블(y_valid)로 분리하기
X_valid = valid_data.iloc[:, 0].values  # 첫 번째 열이 텍스트로 된 입력 특성인 경우
y_valid = valid_data.iloc[:, 1].values  # 두 번째 열이 타깃 레이블인 경우

# 텍스트 데이터를 TF-IDF로 벡터화하기
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_valid = vectorizer.transform(X_valid)

svm_classifier = SVC(kernel='linear')

# 결과를 저장할 파일 열기
with open("models/log/SVM_acc.txt", "w") as file:
    highest_accuracy = 0.0
    best_iteration = 0
    
    for i in range(30):
        # 랜덤 포레스트 분류기 훈련하기
        svm_classifier.fit(X_train, y_train)

        # 검증 데이터에 대한 예측 수행하기
        y_pred = svm_classifier.predict(X_valid)

        # 랜덤 포레스트 분류기 성능 평가하기
        accuracy = accuracy_score(y_valid, y_pred)
        print("정확도:", accuracy)

        # 정확도를 파일에 저장하기
        file.write(f"Iteration {i+1}: Accuracy = {accuracy}\n")
        
        # 정확도 비교
        if accuracy > highest_accuracy:
            highest_accuracy = accuracy
            best_iteration = i+1

    # 가장 높은 정확도를 표시
    file.write(f"\nBest Iteration: {best_iteration}, Highest Accuracy: {highest_accuracy}")