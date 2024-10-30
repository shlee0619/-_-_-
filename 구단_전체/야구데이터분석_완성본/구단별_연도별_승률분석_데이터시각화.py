import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import matplotlib
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px

# 한글 폰트 설정
font_name = matplotlib.font_manager.FontProperties(fname='C:/Windows/Fonts/malgun.ttf').get_name()
matplotlib.rc('font', family=font_name)

# 파일 경로
files = {
    'KIA': r'C:\Mtest\개인연습\KIA_데이터.csv',
    'KT': r'C:\Mtest\개인연습\KT_데이터.csv',
    'LG': r'C:\Mtest\개인연습\LG_데이터.csv',
    'NC': r'C:\Mtest\개인연습\NC_데이터.csv',
    'SSG': r'C:\Mtest\개인연습\SSG_데이터.csv',
    '두산': r'C:\Mtest\개인연습\두산_데이터.csv',
    '롯데': r'C:\Mtest\개인연습\롯데_데이터.csv',
    '삼성': r'C:\Mtest\개인연습\삼성_데이터.csv',
    '키움': r'C:\Mtest\개인연습\키움_데이터.csv',
    '한화': r'C:\Mtest\개인연습\한화_데이터.csv'
}

# 모든 CSV 파일을 로드하여 하나의 DataFrame으로 결합
combined_data = pd.concat([pd.read_csv(file) for file in files.values()], ignore_index=True)

# 팀별 평균 승률 계산 및 시각화
team_avg_win_rate = combined_data.groupby('원정팀')['누적승률'].mean().sort_values(ascending=False).reset_index()

plt.figure(figsize=(10, 6))
plt.barh(team_avg_win_rate['원정팀'], team_avg_win_rate['누적승률'])
plt.title('팀별 평균 승률 비교')
plt.xlabel('평균 승률')
plt.ylabel('원정팀')
plt.show()

# 연도별 팀 승률 추이 시각화
yearly_win_rate = combined_data.groupby(['연도', '원정팀'])['누적승률'].mean().reset_index()
px.line(yearly_win_rate, x='연도', y='누적승률', color='원정팀', title='연도별 팀 승률 추이').show()

# 팀별 이동거리 분포 시각화
px.box(combined_data, x='원정팀', y='누적이동거리', title='팀별 이동거리 분포').show()

# 승률 50% 이상/이하 평균 이동거리 비교
high_win_rate = combined_data[combined_data['누적승률'] > 0.5]
low_win_rate = combined_data[combined_data['누적승률'] <= 0.5]
print("승률 50% 이상의 평균 이동거리:", high_win_rate['누적이동거리'].mean())
print("승률 50% 이하의 평균 이동거리:", low_win_rate['누적이동거리'].mean())

# 원정 거리 구간별 평균 승률 시각화
combined_data['거리구간'] = pd.cut(combined_data['누적이동거리'], bins=10)
distance_win_rate = combined_data.groupby('거리구간')['누적승률'].mean().reset_index()

plt.figure(figsize=(10, 6))
plt.bar(distance_win_rate['거리구간'].astype(str), distance_win_rate['누적승률'])
plt.xticks(rotation=45)
plt.title('원정 거리 구간별 평균 승률')
plt.xlabel('원정 거리 구간')
plt.ylabel('평균 승률')
plt.show()

# 연도별 원정 이동거리와 승률의 상관관계 애니메이션
combined_data['Year'] = combined_data['연도'].astype(str)
px.scatter(combined_data, x='누적이동거리', y='누적승률', color='원정팀', animation_frame='연도', title='원정 이동거리와 승률의 상관관계').show()

# 모든 연도를 포함한 상관관계 애니메이션
combined_data['Year_All'] = combined_data['Year'].replace({year: 'All Years' for year in combined_data['Year'].unique()})
px.scatter(combined_data, x='누적이동거리', y='누적승률', color='원정팀', animation_frame='Year_All', title='원정 이동거리와 승률의 상관관계 (모든 연도 포함)').show()







