import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import matplotlib.pyplot as plt
import seaborn as sns

# 한글 폰트 설정
import matplotlib.font_manager as fm
font_name = fm.FontProperties(fname='C:/Windows/Fonts/malgun.ttf').get_name()
plt.rc('font', family=font_name)

plt.rcParams['axes.unicode_minus'] = False
# 데이터 불러오기
file_path = './data/KT_결합_기록.csv'
dataF = pd.read_csv(file_path)
dataF = dataF[dataF['연도'] != 2024]
dataF = dataF.drop(columns=['승리', '패배', '승률_투수', '도루실패', '피홈런', '순위', '순위_투수', '순위_수비', '연도', '자책점', '실점', '이닝'])
print(dataF)

# 1. 상관관계 행렬 계산
correlation_matrix = dataF.corr()



numeric_data = dataF.select_dtypes(include=['float64', 'int64'])

high_corr_features = correlation_matrix['승률'].sort_values(ascending=False)
# high_corr_features = high_corr_features[high_corr_features > 0.5]


correlation_with_win_rate = numeric_data.corr()['승률'].sort_values(ascending=False)


# 상관관계 결과를 데이터프레임으로 변환하고 인덱스를 열로 변환
correlation_df = correlation_with_win_rate.reset_index()

# 열 이름 지정 (변수 이름과 상관관계 값)
correlation_df.columns = ['변수', '상관관계']

# CSV 파일로 저장 (변수 이름과 상관관계 값 포함)
correlation_df.to_csv('./data/SSG상관관계.csv', index=False, encoding='utf-8-sig')

# 데이터 준비

high_corr_features = correlation_df[correlation_df['상관관계'].abs() > 0.5]

# 1. 바 그래프 (상관관계 높은 변수들)
plt.figure(figsize=(10, 6))
plt.barh(high_corr_features['변수'], high_corr_features['상관관계'], color='skyblue')
plt.xlabel('상관관계')
plt.title('승률과 상관관계가 높은 변수들 (절대값 기준 0.5 이상)')
plt.show()

# 2. 산점도 (변수별 상관관계 시각화)
plt.figure(figsize=(10, 6))
sns.scatterplot(data=high_corr_features, x='변수', y='상관관계', hue='상관관계', palette='coolwarm', s=100)
plt.xticks(rotation=45)
plt.title('승률과 각 변수의 상관관계 (절대값 기준 0.5 이상)')
plt.show()

# 3. 히트맵 (변수들의 상관관계 시각화)
plt.figure(figsize=(8, 6))
sns.heatmap(high_corr_features[['상관관계']].set_index(high_corr_features['변수']).T, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('승률과 상관관계가 높은 변수 히트맵')
plt.show()



sys.exit("종료")

