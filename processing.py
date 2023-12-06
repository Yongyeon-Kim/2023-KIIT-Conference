import json
import os
import csv

output_file = 'datasets/valid/약관/vaildset.csv'

# 폴더 경로 리스트
folder_paths = [
    'datasets/valid/약관/01.유리',
    'datasets/valid/약관/02.불리'
]

# CSV 파일을 추가 모드로 열기 (UTF-8 인코딩)
with open(output_file, 'a', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)

    # 각 폴더 처리
    for folder_path in folder_paths:
        # 폴더 안에 있는 모든 JSON 파일 목록 가져오기
        json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]

        # 각 JSON 파일 처리
        for json_file in json_files:
            file_path = os.path.join(folder_path, json_file)

            # JSON 파일 열기
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

                # 값을 추출하여 CSV 파일에 추가하기
                clause_article = data['clauseArticle']
                dv_antageous = data['dvAntageous']
                writer.writerow([clause_article, dv_antageous])
