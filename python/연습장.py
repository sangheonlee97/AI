import pandas as pd

# XLSX 파일 읽기
xlsx_file_path = 'KETI-2017-SL-Annotation-v2_1.xlsx'
df = pd.read_excel(xlsx_file_path)

# CSV 파일로 저장
csv_file_path = 'KETI-2017-SL-Annotation-v2_1.csv'
df.to_csv(csv_file_path, index=False)