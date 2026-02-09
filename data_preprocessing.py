import json
import os
import csv
import pandas as pd

def create_dataset(data_path, output_file):
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['text', 'label'])

        for label_type in ['01.유리', '02.불리']:
            folder_path = os.path.join(data_path, label_type)
            if not os.path.isdir(folder_path):
                continue
            
            label = 1 if label_type == '01.유리' else 0
            
            json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]

            for json_file in json_files:
                file_path = os.path.join(folder_path, json_file)

                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    text = data['clauseArticle']
                    # Ensure text is a single string
                    if isinstance(text, list):
                        text = ' '.join(map(str, text))
                        
                    writer.writerow([text, label])

# Create train and validation datasets
create_dataset('datasets/train/약관', 'datasets/train.csv')
create_dataset('datasets/valid/약관', 'datasets/valid.csv')

print("train.csv and valid.csv created successfully.")

