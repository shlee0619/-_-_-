import pandas as pd
import matplotlib.pyplot as plt
import re
import matplotlib.font_manager as fm
font_name = fm.FontProperties(fname='C:/Windows/Fonts/malgun.ttf').get_name()
plt.rc('font', family=font_name)

# # 데이터 불러오기
file_path = './data/관중데이터.csv'
dataF = pd.read_csv(file_path)
dataF['관중 데이터'] = dataF['관중 데이터'].str.replace(',', '')
dataF.to_csv('./data/관중데이터.csv', index=False, encoding='utf-8-sig')
dataF = pd.read_csv(file_path, encoding='utf-8')
# 1. 연도를 내림차순으로 정렬
dataF = dataF.sort_values(by='연도', ascending=False)

# 2. 각 팀별 관중 데이터에서 괄호와 괄호 안의 내용을 제거하는 함수
def remove_parentheses(text):
    return re.sub(r'\([^)]*\)', '', str(text))

# 괄호 제거를 적용할 특정 열 (예: '관중 데이터' 열)


names = ['삼성','KIA','롯데','LG','두산','한화','SSG','키움','NC','KT']

for i in names:
    dataF[i] = dataF[i].apply(remove_parentheses)

# 3. 데이터 저장 (CSV 파일을 다시 저장)
dataF.to_csv('./data/관중데이터_cleaned.csv', index=False, encoding='utf-8-sig')

# 결과 확인
print(dataF.head())

dataF = dataF.sort_values(by='연도', ascending=True)

# 4. 데이터 저장 (CSV 파일을 다시 저장)
dataF.to_csv('./data/관중데이터.csv', index=False, encoding='utf-8-sig')

data_melted = pd.melt(dataF, id_vars=['연도'], var_name='팀명', value_name='관중 데이터')

# 결과 확인
print(data_melted.head())

#변환된 데이터를 저장
data_melted.to_csv('./data/관중데이터.csv', index=False, encoding='utf-8-sig')
file_path = './data/연도별평균연봉.csv'
dataF = pd.read_csv(file_path, encoding='cp949')
dataF.to_csv('./data/연도별평균연봉.csv', index=False, encoding='utf-8-sig')


