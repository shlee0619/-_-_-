import pandas as pd
import glob
import os


folder_path = 'C:\\Mtest\\개인연습'


csv_files = glob.glob(os.path.join(folder_path, '*.csv'))


for file_path in csv_files:
    
    data = pd.read_csv(file_path)
    
    # 각 홈구장이나 구단의 이름이 변경된 경우, 교체
    data.replace("넥센", "키움", inplace=True)
    data.replace("SK", "SSG", inplace=True)
    data.replace("울산", "사직", inplace=True)
    data.replace("청주", "대전", inplace=True)
    data.replace("마산", "창원", inplace=True)
    data.replace("포항", "대구", inplace=True)
    data.replace("목동", "고척", inplace=True)
    #파일 저장
    data.to_csv(file_path, index=False, encoding='utf-8-sig')
    print(f"Updated and saved file: {file_path}")

print("All files processed successfully.")