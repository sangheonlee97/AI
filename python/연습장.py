import pandas as pd

# XLSX 파일 읽기
xlsx_file_path = '한국마사회_경주상세정보(제주_부경)_20230720.xlsx'
df = pd.read_excel(xlsx_file_path)

# CSV 파일로 저장
csv_file_path = '경마.csv'
df.to_csv(csv_file_path, index=False)